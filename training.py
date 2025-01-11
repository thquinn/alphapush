import math
import time
import torch
import torch.nn as nn
import uuid
from mcts import MCTS
from network import NullNet
from pushfight import PFPiece, PFState
from test_null_training import NullTrainingNetwork

@torch.no_grad()
def generate_training_data(net, min_training_items, evals_per_position=2048, parallelism=1):
    net.eval()
    training_inputs = []
    training_outputs = []
    mctses = [None] * parallelism
    root_output = net.forward(PFState().to_tensor()).numpy()[0]
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
            # Get inferences on all active positions, update MCTS instances.
            if isinstance(net, NullNet):
                output = None
            else:
                input_tensor = torch.stack([mcts.get_current_state_tensor() for mcts in mctses])
                output = net.forward(input_tensor).numpy()
            for i, mcts in enumerate(mctses):
                mcts.receive_network_output(output[i] if output is not None else None)
        for mcts in mctses:
            mcts.values.append(1 if mcts.root.state.white_to_move else -1)
            mcts.policies.append(mcts.to_policy_tensor())
            mcts.advance_root(temperature=0.25)
            if mcts.root.state.winner != PFPiece.Empty:
                # Construct training outputs.
                training_values = mcts.values
                training_policies = mcts.policies
                if mcts.root.state.winner == PFPiece.Black:
                    training_values = [-x for x in training_values]
                training_inputs.extend([state.to_tensor() for state in mcts.history])
                assert len(training_values) == len(training_policies)
                training_outputs.extend([[training_values[i]] + training_policies[i] for i in range(len(training_values))])
                assert len(training_inputs) == len(training_outputs)
    training_inputs = torch.stack(training_inputs)
    training_outputs = torch.tensor(training_outputs)
    return training_inputs, training_outputs

@torch.no_grad()
def eval_match(old_net, new_net, games=100, evals_per_position=512, verbose=False):
    old_net.eval()
    new_net.eval()
    new_wins = 0
    old_white_wins = 0
    new_white_wins = 0
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
            if root_output is not None:
                root_output = root_output[0].numpy()
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
            if new_plays_white:
                new_white_wins += 1
            else:
                old_white_wins += 1
    print(f'Finished {games} {evals_per_position}-eval games in {math.floor(time.time() - start_time)} seconds.')
    print(f'New network won {new_wins} of {games} ({(new_wins / games * 100):.2f}%). {len(game_hashes)} unique games. Average length {(total_moves / games):.2f}. New won {new_white_wins}/{games // 2} as white, old won {old_white_wins}/{games // 2}.')
    return new_wins / games >= .55

def generate_training_dataset():
    model_name = '270K_v001'
    net = torch.load(f'model_{model_name}.pt', weights_only=False).cpu()
    inputs = []
    outputs = []
    print('Starting dataset generation.')
    while True:
        batch_time = time.time()
        input, output = generate_training_data(net, min_training_items=10000, parallelism=32)
        print(f'Generated {input.shape[0]} examples in {(time.time() - batch_time):.1f} seconds. Saving...')
        inputs.append(input)
        outputs.append(output)
        torch.save({'X': torch.cat(inputs, dim=0), 'Y': torch.cat(outputs, dim=0)}, f'dataset_{model_name}_1K_{uuid.uuid4()}.tnsr')
        print('Saved.')

if __name__ == '__main__':
    generate_training_dataset()