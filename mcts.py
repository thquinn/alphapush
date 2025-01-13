import math
import random

import numpy as np
from pushfight import PFMove, PFPiece, PFState
import torch
import torch.nn as nn

class MCTSNode:
    EXPLORATION = 1.1

    parent: 'MCTSNode'
    children: dict[PFMove, 'MCTSNode']
    state: PFState
    visits: int
    total_value: float
    child_policies = list[float]

    def __init__(self, parent: 'MCTSNode', state: PFState):
        self.parent = parent
        self.children = {move: None for move in state.moves} if state.winner == PFPiece.Empty else {}
        self.state = state
        self.visits = 0
        self.total_value = 0
    
    def average_reward(self):
        return self.total_value / self.visits
    
    def get_upper_bound(self, policy):
        return self.average_reward() + math.exp(policy) * MCTSNode.EXPLORATION * math.sqrt(self.parent.visits) / (1 + self.visits)
    
    def debug_print(self, depth=1, current_depth=1, top_n=3):
        moves = [move for move in self.children if self.children[move] is not None]
        expected_values = [f'{self.children[move].average_reward():.2f}v' for move in moves]
        policy_percents = [f'{self.child_policies[int(move)]:.2f}p' for move in moves]
        prints = list(zip(moves, [self.children[move].visits for move in moves], expected_values, policy_percents))
        prints.sort(key=lambda p: p[1], reverse=True)
        for i in range(min(top_n, len(prints))):
            print(f'{"  " * (current_depth - 1)}{prints[i]}')
            if current_depth < depth:
                self.children[prints[i][0]].debug_print(depth, current_depth + 1, top_n)
        if len(prints) > top_n and top_n > 1:
            if len(prints) > top_n + 1:
                print(f'{" " * (current_depth - 1)}...')
            print(f'{" " * (current_depth - 1)}{prints[-1]}')

class MCTS:
    root: MCTSNode
    current_node: MCTSNode
    history: list[PFState]
    values: list[float]
    policies: list[list[float]]

    def __init__(self, root_state, root_output):
        self.root = MCTSNode(None, root_state)
        self.root.visits = 0
        self.current_node = self.root
        self.receive_network_output(root_output)
        self.history = []
        self.values = []
        self.policies = []
    
    def run_with_net(self, net, min_evals, advance=True, temperature=0.1, print_depth=0, top_n=0):
        evals = 0
        while evals < min_evals or not all(self.root.children.values()):
            self.select_and_expand()
            output = net.forward(self.get_current_state_tensor())
            if output is not None:
                output = output[0].numpy()
            self.receive_network_output(output)
            evals += 1
        if advance:
            self.advance_root(temperature=temperature, print_depth=print_depth, top_n=top_n)
    
    def select_and_expand(self):
        assert self.current_node == self.root
        # If the root is a winning state, there's no work to be done.
        if self.current_node.state.winner != PFPiece.Empty:
            return
        for _ in range(10):
            # Select until we find a node with missing children.
            while self.current_node.state.winner == PFPiece.Empty and all(self.current_node.children.values()):
                max_move = max(self.current_node.children.keys(), key=lambda k: self.current_node.children[k].get_upper_bound(self.current_node.child_policies[int(k)]))
                self.current_node = self.current_node.children[max_move]
            # If the root is a winning state, backpropagate and repeat.
            if self.current_node.state.winner != PFPiece.Empty:
                self.receive_network_output(None)
                continue
            # Otherwise, select a new child and prepare for network output.
            if not all(self.current_node.children.values()):
                missing_move = None
                for k, v in self.current_node.children.items():
                    if v is None:
                        missing_move = k
                        break
                new_state = PFState.copy(self.current_node.state)
                new_state.move(missing_move)
                new_node = MCTSNode(self.current_node, new_state)
                self.current_node.children[missing_move] = new_node
                self.current_node = new_node
            return

    def get_current_state_tensor(self):
        return self.current_node.state.to_tensor()
    
    zero_policy = [0] * 806
    def receive_network_output(self, output):
        # Set policy.
        if self.current_node.state.winner == PFPiece.Empty:
            if output is None:
                self.current_node.child_policies = self.zero_policy
            else:
                policy = output[1:]
                assert len(policy) == 806
                self.current_node.child_policies = policy
        # Backpropagate value up the tree.
        value = 0 if output is None else output[0]
        if self.current_node.parent:
            current_white_to_move = self.current_node.parent.state.white_to_move
            current_player = PFPiece.White if current_white_to_move else PFPiece.Black
        if self.current_node.state.winner != PFPiece.Empty:
            value = 1 if (self.current_node.state.winner == current_player) else -1
        while self.current_node is not None:
            self.current_node.visits += 1
            if self.current_node.parent:
                self.current_node.total_value += value if (self.current_node.parent.state.white_to_move == current_white_to_move) else -value
            self.current_node = self.current_node.parent
        self.current_node = self.root
    
    def to_policy_tensor(self):
        assert self.current_node == self.root
        policy = [0] * 806
        child_sum = sum([child.visits for child in self.root.children.values()])
        for move, child in self.root.children.items():
            if child is not None:
                policy[int(move)] = child.visits / child_sum
        return policy
    
    def advance_root(self, temperature=1, print_depth=0, top_n=3):
        self.history.append(self.root.state)
        moves = list(self.root.children.keys())
        if print_depth > 0:
            self.root.debug_print(print_depth, top_n=top_n)
        if temperature == 0:
            move = max(self.root.children, key=lambda k: self.root.children[k].visits)
        else:
            weights = np.array([child.visits / (self.root.visits - 1) for child in self.root.children.values()])
            if temperature != 1:
                weights = weights ** (1 / temperature)
                weights /= weights.sum()
            move = np.random.choice(moves, p=weights)
        if print_depth > 0:
            print(f'Chose {move} with {self.root.children[move].visits} visits.\n')
        self.root = self.root.children[move]
        self.root.parent = None
        self.current_node = self.root
        return move
