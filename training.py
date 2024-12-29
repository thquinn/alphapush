import torch
import torch.nn as nn
from mcts import MCTS
from pushfight import PFPiece, PFState

def generate_training_data(net, min_training_items=128, evals_per_position=128, parallelism=1):
    net.eval()
    training_inputs = []
    training_outputs = []
    mctses = [None] * parallelism
    root_output = net.forward(PFState().to_tensor())
    while len(training_outputs) < min_training_items:
        for i, mcts in enumerate(mctses):
            if mcts is None or mcts.root.state.winner != PFPiece.Empty:
                mctses[i] = MCTS(PFState(), root_output)
        for eval in range(evals_per_position):
            # Gather input vectors from all MCTS instances.
            for mcts in mctses:
                mcts.select_and_expand()
            input_tensor = torch.stack([mcts.get_current_state_tensor() for mcts in mctses])
            # Get inferences on all active positions, update MCTS instances.
            output = net.forward(input_tensor)
            log_probs = nn.functional.softmax(output[:,1:], dim=1)
            for i, mcts in enumerate(mctses):
                value = output[i, 0].item()
                policy = log_probs[i,:].tolist()
                mcts.receive_value_and_policy(value, policy)
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
                print(f'Self-play game finished. Batch contains {len(training_inputs)} examples.')
    training_inputs = torch.stack(training_inputs)
    training_outputs = torch.tensor(training_outputs)
    return training_inputs, training_outputs

def eval_match(old_net, new_net, games=40, evals_per_position=128):
    new_wins = 0
    for game in range(games):
        print(f'Starting evaluation game {game + 1} of {games}.')
        state = PFState()
        new_plays_white = game % 2 == 0
        white, black = (new_net, old_net) if new_plays_white else (old_net, new_net)
        while state.winner == PFPiece.Empty:
            net = white if state.white_to_move else black
            root_output = white(state.to_tensor())
            mcts = MCTS(state, root_output)
            mcts.run_with_net(net, evals_per_position)
            state = mcts.root.state
            print()
            print(state)
        print()
        if (state.winner == PFPiece.White) == new_plays_white:
            new_wins += 1
            print('New network wins.')
        else:
            print('Old network wins.')
    print(f'New network won {new_wins} of {games} games ({(new_wins / games * 100):.2f}%).')
    return new_wins / games >= .55