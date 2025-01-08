import math
from multiprocessing import Manager, Pool
import os
import random
import time
import numpy as np
import torch

from mcts import MCTS
from network import NullNet
from pushfight import PFPiece, PFState

def worker(q):
    print(f"Worker started with PID: {os.getpid()}")
    try:
        while True:
            result = play_nullnet_game()
            q.put(result)
    except Exception as e:
        print(f"Worker {os.getpid()} failed with error: {e}")
        raise e

def generate_nullnet_data(amount=int(1e6)):
    start_time = time.time()
    with Manager() as manager:
        queue = manager.Queue()
        with Pool() as pool:
            print(f'Started a pool of {pool._processes} workers.')
            pool.starmap_async(worker, [(queue,)] * pool._processes)
            inputs = []
            outputs = []
            example_count = 0
            while example_count < amount:
                input, output = queue.get()
                inputs.append(input)
                outputs.append(output)
                old_example_count = example_count
                example_count += input.shape[0]
                if example_count % 1000 < old_example_count % 1000:
                    print(f'Generated {example_count}/{amount} training examples...')
                if example_count % 10000 < old_example_count % 10000:
                    print(f'Saving progress...')
                    torch.save({'X': torch.cat(inputs, dim=0), 'Y': torch.cat(outputs, dim=0)}, f'nullset_{math.floor(start_time)}.tnsr')
                    print('Saved.')
            pool.terminate()
            print(f'Done!')

net = NullNet()
def play_nullnet_game():
    state = PFState()
    for _ in range(8):
        state.move(random.choice(state.moves))
    mcts = MCTS(state, None)
    while mcts.root.state.winner == PFPiece.Empty:
        mcts.values.append(1 if mcts.root.state.white_to_move else -1)
        mcts.run_with_net(net, 2048, advance=False)
        mcts.policies.append(mcts.to_policy_tensor())
        mcts.advance_root(temperature=.25)
    values = mcts.values
    if mcts.root.state.winner == PFPiece.Black:
        values = [-x for x in values]
    values = torch.tensor(values)
    policies = torch.tensor(mcts.policies)
    outputs = torch.cat((values.unsqueeze(1), policies), dim=1).float()
    inputs = torch.stack([state.to_tensor() for state in mcts.history]).float()
    return (inputs, outputs)

if __name__ == '__main__':
    generate_nullnet_data()