import math
import random

import numpy as np
from pushfight import PFMove, PFPiece, PFState
import torch
import torch.nn as nn

class MCTSNode:
    EXPLORATION = 5

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
        return self.average_reward() + policy * MCTSNode.EXPLORATION * math.sqrt(self.parent.visits) / (1 + self.visits)

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
    
    def run_with_net(self, net, min_evals):
        evals = 0
        while evals < min_evals or not all(self.root.children.values()):
            self.select_and_expand()
            input_tensor = self.get_current_state_tensor()
            output = net.forward(input_tensor)
            self.receive_network_output(output)
            evals += 1
        self.advance_root(temperature=.25)
    
    def select_and_expand(self):
        while self.current_node.state.winner == PFPiece.Empty and all(self.current_node.children.values()):
            max_move = max(self.current_node.children, key=lambda k: self.current_node.children[k].get_upper_bound(self.current_node.child_policies[int(k)]))
            self.current_node = self.current_node.children[max_move]
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

    def get_current_state_tensor(self):
        return self.current_node.state.to_tensor()
    
    def receive_network_output(self, output):
        if self.current_node.state.winner == PFPiece.Empty:
            policy = output[1:]
            assert len(policy) == 806
            if not self.current_node.state.white_to_move:
                # The network saw the board in reverse order, so now we reverse each component of the network output.
                policy = torch.cat((
                    torch.flip(policy[:26], [0]),
                    torch.flip(policy[26:26+26*26], [0]),
                    torch.flip(policy[-26*4:], [0]),
                ))
            # Set policy.
            policy_mask = np.ones(806, dtype=bool)
            assert len(self.current_node.children) > 0
            for move in self.current_node.children.keys():
                policy_mask[int(move)] = False
            policy = policy.masked_fill(torch.from_numpy(policy_mask), float('-inf'))
            policy = nn.functional.softmax(policy, dim=0).tolist()
            self.current_node.child_policies = policy
        # Backpropagate value up the tree.
        value = output[0].item()
        if self.current_node.state.winner != PFPiece.Empty:
            value = 1 if self.current_node.state.winner == PFPiece.White else -1
        while self.current_node is not None:
            self.current_node.visits += 1
            if self.current_node.parent:
                self.current_node.total_value += value if self.current_node.parent.state.white_to_move else -value
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
    
    def advance_root(self, temperature=1, debug_print=False):
        self.history.append(self.root.state)
        moves = list(self.root.children.keys())
        if debug_print:
            expected_values = [f'{child.average_reward():.2f}' for child in self.root.children.values()]
            policy_percents = [f'{(self.root.child_policies[int(move)] * 100):.2f}%' for move in self.root.children.keys()]
            prints = list(zip(moves, [child.visits for child in self.root.children.values()], expected_values, policy_percents))
            prints.sort(key=lambda p: p[1], reverse=True)
            print(prints)
        if temperature == 0:
            move = max(self.root.children, key=lambda k: self.root.children[k].visits)
        else:
            weights = np.array([child.visits / (self.root.visits - 1) for child in self.root.children.values()])
            if temperature != 1:
                weights = weights ** (1 / temperature)
                weights /= weights.sum()
            move = np.random.choice(moves, p=weights)
        if debug_print:
            print(move)
        self.root = self.root.children[move]
        self.root.parent = None
        self.current_node = self.root
