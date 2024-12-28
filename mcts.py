import math
import random

import numpy as np
from pushfight import PFMove, PFPiece, PFState
import torch
import torch.nn as nn

class MCTSNode:
    EXPLORATION = math.sqrt(2)

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
    
    def get_upper_bound(self, policy):
        return self.total_value / self.visits + policy * MCTSNode.EXPLORATION * math.sqrt(math.log(self.parent.visits) / self.visits)

class MCTS:
    root: MCTSNode
    current_node: MCTSNode
    history: list[PFState]
    values: list[float]
    policies: list[list[float]]

    def __init__(self, root_state, root_output):
        self.root = MCTSNode(None, root_state)
        self.root.visits = 1
        root_policies = [tensor.item() for tensor in root_output[1:].data]
        self.root.child_policies = root_policies
        self.current_node = self.root
        self.history = []
        self.values = []
        self.policies = []
    
    def run_with_net(self, net, min_evals):
        evals = 0
        while evals < min_evals or not all(self.root.children.values()):
            self.select_and_expand()
            input_tensor = self.get_current_state_tensor()
            output = net.forward(input_tensor)
            log_probs = nn.functional.softmax(output[1:], dim=0)
            self.receive_value_and_policy(output[0].item(), [tensor.item() for tensor in log_probs.data])
            evals += 1
        self.advance_root()
    
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
    
    def receive_value_and_policy(self, value, policy):
        assert len(policy) == 806
        if not self.current_node.state.white_to_move:
            # The network saw the board in reverse order, so now we reverse each component of the network output.
            value = -value
            policy[:26] = policy[:26][::-1]
            policy[26:26+26*26] = policy[26:26+26*26][::-1]
            policy[-26*4:] = policy[-26*4:][::-1]
        self.current_node.child_policies = policy
        if self.current_node.state.winner != PFPiece.Empty:
            color = PFPiece.White if self.current_node.state.white_to_move else PFPiece.Black
            value = 1 if color == self.current_node.state.winner else -1
        while self.current_node is not None:
            self.current_node.visits += 1
            total_child_visits = sum([child.visits for child in self.current_node.children.values() if child is not None])
            if self.current_node.parent:
                self.current_node.total_value += value if self.current_node.parent.state.white_to_move else -value
            self.current_node = self.current_node.parent
        self.current_node = self.root
    
    def to_policy_tensor(self):
        assert self.current_node == self.root
        policy = [0] * 806
        for move, child in self.root.children.items():
            if child is not None:
                policy[int(move)] = child.visits / self.root.visits
        return policy

    def to_legality_mask(self):
        assert len(self.root.state.moves) > 0
        legality = [0] * 806
        for move in self.root.state.moves:
            legality[int(move)] = 1
        return legality
    
    def advance_root(self):
        self.history.append(self.root.state)
        moves = list(self.root.children.keys())
        weights = [child.visits / (self.root.visits - 1) for child in self.root.children.values()]
        move = np.random.choice(moves, p=weights)
        self.root = self.root.children[move]
        self.root.parent = None
        self.current_node = self.root
