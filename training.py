import math
import time
import torch
import torch.nn as nn
from mcts import MCTS
from pushfight import PFPiece, PFState

def generate_training_data(net, min_training_items, evals_per_position=1024, parallelism=1):
    net.eval()
    training_inputs = []
    training_outputs = []
    mctses = [None] * parallelism
    root_output = net.forward(PFState().to_tensor())
    while len(training_outputs) < min_training_items:
        for i, mcts in enumerate(mctses):
            if mcts is None or mcts.root.state.winner != PFPiece.Empty or len(mcts.history) > 200:
                if mcts and len(mcts.history) > 200:
                    for state in mcts.history[-6:]:
                        print(state)
                    print('Game hit 200 moves. Restarting...')
                mctses[i] = MCTS(PFState(), root_output)
        for eval in range(evals_per_position):
            # Gather input vectors from all MCTS instances.
            for mcts in mctses:
                mcts.select_and_expand()
            input_tensor = torch.stack([mcts.get_current_state_tensor() for mcts in mctses])
            # Get inferences on all active positions, update MCTS instances.
            output = net.forward(input_tensor)
            for i, mcts in enumerate(mctses):
                mcts.receive_network_output(output[i])
        for mcts in mctses:
            mcts.values.append(1 if mcts.root.state.white_to_move else -1)
            mcts.policies.append(mcts.to_policy_tensor())
            mcts.advance_root()
            if mcts.root.state.winner != PFPiece.Empty:
                # Construct training outputs.
                training_values = mcts.values
                training_policies = mcts.policies
                if mcts.root.state.winner == PFPiece.Black:
                    training_values = [-x for x in training_values]
                training_inputs.extend([state.to_tensor() for state in mcts.history])
                training_outputs.extend([[training_values[i]] + training_policies[i] for i in range(len(training_values))])
                assert len(training_inputs) == len(training_outputs)
    training_inputs = torch.stack(training_inputs)
    training_outputs = torch.tensor(training_outputs)
    return training_inputs, training_outputs

def eval_match(old_net, new_net, games=100, evals_per_position=256, verbose=False):
    new_wins = 0
    white_wins = 0
    game_hashes = set()
    total_moves = 0
    start_time = time.time()
    for game in range(games):
        if verbose:
            print(f'Starting evaluation game {game + 1} of {games}.')
        state = PFState()
        new_plays_white = game % 2 == 0
        white, black = (new_net, old_net) if new_plays_white else (old_net, new_net)
        state_hashes = []
        while state.winner == PFPiece.Empty:
            state_hashes.append(hash(state))
            total_moves += 1
            net = white if state.white_to_move else black
            root_output = net.forward(state.to_tensor())
            mcts = MCTS(state, root_output)
            mcts.run_with_net(net, evals_per_position)
            state = mcts.root.state
            if verbose:
                print()
                print(state)
        game_hashes.add(tuple(state_hashes + [hash(state)]))
        if (state.winner == PFPiece.White) == new_plays_white:
            new_wins += 1
            if verbose:
                print('\nNew network wins.')
        else:
            if verbose:
                print('\nOld network wins.')
        if state.winner == PFPiece.White:
            white_wins += 1
    print(f'Finished {games} {evals_per_position}-eval games in {math.floor(time.time() - start_time)} seconds.')
    print(f'New network won {new_wins} of {games} ({(new_wins / games * 100):.2f}%). {len(game_hashes)} unique games. Average length {(total_moves / games):.2f}. White won {white_wins}/{games}.')
    return new_wins / games >= .55