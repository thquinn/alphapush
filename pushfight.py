from collections import deque
from dataclasses import dataclass
from enum import Enum, IntFlag
from itertools import chain
import numpy as np
import torch

class PFPiece(IntFlag):
    Empty = 0,
    White = 1,
    Black = 2,
    Pusher = 4,

    def __repr__(self):
        if self == PFPiece.Empty:
            return '.'
        if PFPiece.White in self:
            return 'W' if PFPiece.Pusher in self else 'w'
        return 'B' if PFPiece.Pusher in self else 'b'
    def __str__(self): return self.__repr__()

# Bitwise operations between IntFlags was consuming like 15% of self-play time... why are we all using Python, again?
white_pusher = PFPiece.White | PFPiece.Pusher
black_pusher = PFPiece.Black | PFPiece.Pusher

class PFDirection(Enum):
    Up = 0
    Left = 1
    Right = 2
    Down = 3
    NoPush = 4

@dataclass
class PFMove:
    placeIndex: int
    moveFromIndex: int
    moveToIndex: int
    pushFromIndex: int
    pushDirection: PFDirection

    @staticmethod
    def place(index: int):
        return PFMove(index, -1, -1, -1, PFDirection.NoPush)
    @staticmethod
    def move(fromIndex: int, toIndex: int):
        return PFMove(-1, fromIndex, toIndex, -1, PFDirection.NoPush)
    @staticmethod
    def push(fromIndex: int, direction: PFDirection):
        return PFMove(-1, -1, -1, fromIndex, direction)
    @staticmethod
    def parse(s):
        try:
            tokens = s.lower().replace('@', '.').replace('>', '.').split('.')
            index_one = int(tokens[1])
            if tokens[0] == 'place':
                return PFMove.place(index_one)
            if tokens[0] == 'move':
                index_two = int(tokens[2])
                return PFMove.move(index_one, index_two)
            if tokens[0] == 'push':
                direction = PFDirection[tokens[2].capitalize()]
                return PFMove.push(index_one, direction)
        except Exception as e:
            print(e)
            pass
        return None
    
    def __int__(self):
        if self.placeIndex >= 0:
            return self.placeIndex
        if self.moveFromIndex >= 0:
            return 26 + self.moveFromIndex * 26 + self.moveToIndex
        return 26 + 26 * 26 + self.pushFromIndex * 4 + self.pushDirection.value
    def __hash__(self): return int(self)
    
    def __repr__(self):
        if self.placeIndex >= 0:
            return f'Place@{self.placeIndex}'
        if self.moveFromIndex >= 0:
            return f'Move@{self.moveFromIndex}>{self.moveToIndex}'
        return f'Push@{self.pushFromIndex}.{self.pushDirection.name}'
    def __str__(self): return self.__repr__()

# Instantiating these PFMoves was consuming like 20% of self-play time... why are we all using Python, again?
staticmove_places = [PFMove.place(i) for i in range(26)]
staticmove_moves = [[PFMove.move(x, y) for y in range(26)] for x in range(26)]
staticmove_pushes = [[PFMove.push(i, dir) for dir in [PFDirection.Up, PFDirection.Left, PFDirection.Right, PFDirection.Down]] for i in range(26)]

class PFState:
    #  (white side)
    #      0  1
    #  |2  3  4
    #  |5  6  7  8 |
    #  |9  10 11 12|
    #  |13 14 15 16|
    #  |17 18 19 20|
    #      21 22 23|
    #      24 25
    # (black side)
    DIRECTIONS = [PFDirection.Up, PFDirection.Left, PFDirection.Right, PFDirection.Down]
    WALL = -1
    VOID = -2
    NEIGHBORS = [
        [VOID, VOID, 1, 3], [VOID, 0, VOID, 4], [VOID, WALL, 3, 5], [0, 2, 4, 6], [1, 3, VOID, 7], [2, WALL, 6, 9], [3, 5, 7, 10], [4, 6, 8, 11], [VOID, 7, WALL, 12], [5, WALL, 10, 13], # 0-9
        [6, 9, 11, 14], [7, 10, 12, 15], [8, 11, WALL, 16], [9, WALL, 14, 17], [10, 13, 15, 18], [11, 14, 16, 19], [12, 15, WALL, 20], [13, WALL, 18, VOID], [14, 17, 19, 21], [15, 18, 20, 22], # 10-19
        [16, 19, WALL, 23], [18, VOID, 22, 24], [19, 21, 23, 25], [20, 22, WALL, VOID], [21, VOID, 25, VOID], [22, 24, VOID, VOID] # 20-25
    ]
    PASSABLE_NEIGHBORS = [[b for b in a if b >= 0] for a in NEIGHBORS]

    winner: PFPiece
    moves: list[PFMove]
    board: list[PFPiece]
    num_pieces: int
    white_to_move: bool
    moves_left: int
    anchor_position: int

    def __init__(self, set_moves=True):
        self.winner = PFPiece.Empty
        self.board = [PFPiece.Empty] * 26
        self.num_pieces = 0
        self.white_to_move = True
        self.moves_left = 0
        self.anchor_position = -1
        if set_moves:
            self.set_moves()
    @classmethod
    def copy(cls, original: 'PFState'):
        copy = PFState(set_moves=False)
        copy.winner = original.winner
        copy.board = original.board.copy()
        copy.num_pieces = original.num_pieces
        copy.white_to_move = original.white_to_move
        copy.moves_left = original.moves_left
        copy.anchor_position = original.anchor_position
        copy.moves = original.moves
        return copy
    @classmethod
    def construct(cls, board_string: str, white_to_move: bool, moves_left: int, anchor_position: int):
        assert len(board_string) == 26
        state = PFState(set_moves=False)
        for i, char in enumerate(board_string):
            if char == 'w':
                state.board[i] = PFPiece.White
            elif char == 'W':
                state.board[i] = PFPiece.White | PFPiece.Pusher
            elif char == 'b':
                state.board[i] = PFPiece.Black
            elif char == 'B':
                state.board[i] = PFPiece.Black | PFPiece.Pusher
        state.num_pieces = len([p for p in state.board if p != PFPiece.Empty])
        state.white_to_move = white_to_move
        state.moves_left = moves_left
        state.anchor_position = anchor_position
        state.set_moves()
        return state

    def set_moves(self):
        if self.num_pieces < 10:
            self.moves = [staticmove_places[i] for i in (range(0, 13) if self.white_to_move else range(13, 26)) if self.board[i] == PFPiece.Empty]
            return
        self.moves = []
        if self.winner != PFPiece.Empty:
            return
        # Pushing moves.
        color = PFPiece.White if self.white_to_move else PFPiece.Black
        my_pusher = white_pusher if self.white_to_move else black_pusher
        for i in range(26):
            if self.board[i] == (color | PFPiece.Pusher):
                for direction in PFState.DIRECTIONS:
                    if self.can_push(i, direction):
                        self.moves.append(staticmove_pushes[i][direction.value])
        # Moving... moves.
        if self.moves_left > 0:
            for i in range(26):
                if self.board[i] == color or self.board[i] == my_pusher:
                    queue = deque([i])
                    seen = [False] * 26
                    seen[i] = True
                    while len(queue) > 0:
                        current = queue.popleft()
                        for neighbor in PFState.PASSABLE_NEIGHBORS[current]:
                            if seen[neighbor]:
                                continue
                            if self.board[neighbor] == PFPiece.Empty:
                                self.moves.append(staticmove_moves[i][neighbor])
                                seen[neighbor] = True
                                queue.append(neighbor)
    
    def can_push(self, index: int, direction: PFDirection):
        index = PFState.NEIGHBORS[index][direction.value]
        if index < 0:
            return False
        if self.board[index] == PFPiece.Empty:
            return False
        while True:
            if index == self.anchor_position:
                return False
            index = PFState.NEIGHBORS[index][direction.value]
            if index == PFState.WALL:
                return False
            if index == PFState.VOID:
                return True
            if self.board[index] == PFPiece.Empty:
                return True
    
    def get_move_winner(self, move: PFMove):
        if move.pushFromIndex == -1:
            return PFPiece.Empty
        index = move.pushFromIndex
        last_piece_color = None
        while True:
            index = PFState.NEIGHBORS[index][move.pushDirection.value]
            if index == PFState.VOID:
                assert last_piece_color is not None
                return PFPiece.Black if last_piece_color == PFPiece.White else PFPiece.White
            if self.board[index] == PFPiece.Empty:
                return PFPiece.Empty
            last_piece_color = PFPiece.White if self.board[index] & PFPiece.White else PFPiece.Black
    
    def move(self, move: PFMove, set_moves=True):
        if move.placeIndex != -1:
            assert self.num_pieces < 10
            self.board[move.placeIndex] = (PFPiece.White if self.white_to_move else PFPiece.Black) | (PFPiece.Pusher if self.num_pieces % 5 < 3 else 0)
            self.num_pieces += 1
            self.white_to_move = self.num_pieces < 5 or self.num_pieces == 10
            if self.num_pieces == 10:
                self.moves_left = 2
        elif move.moveFromIndex != -1:
            assert self.moves_left > 0
            self.board[move.moveToIndex] = self.board[move.moveFromIndex]
            self.board[move.moveFromIndex] = PFPiece.Empty
            self.moves_left -= 1
        else:
            assert move.pushFromIndex > -1
            index = move.pushFromIndex
            value = self.board[index]
            self.board[index] = PFPiece.Empty
            while value != PFPiece.Empty:
                next_index = PFState.NEIGHBORS[index][move.pushDirection.value]
                if next_index < 0:
                    if next_index == PFState.VOID:
                        self.winner = PFPiece.White if value & PFPiece.Black else PFPiece.Black
                    break
                next_value = self.board[next_index]
                self.board[next_index] = value
                value = next_value
                index = next_index
            self.anchor_position = PFState.NEIGHBORS[move.pushFromIndex][move.pushDirection.value]
            assert self.anchor_position >= 0
            if self.winner != PFPiece.Empty:
                self.moves = []
                return
            self.white_to_move = not self.white_to_move
            self.moves_left = 2
        if set_moves:
            self.set_moves()
        if len(self.moves) == 0:
            self.winner = PFPiece.Black if self.white_to_move else PFPiece.White
    
    @torch.no_grad()
    def to_tensor(self):
        arr = np.zeros(160)
        # 3 elements: one-hot, are we [placing pushers], [placing round pieces], or [playing the game]?
        if self.num_pieces < 10:
            arr[0 if self.num_pieces % 5 < 3 else 1] = 1
        else:
            arr[2] = 1
        # 1 element: how many moves do we have before we have to push?
        arr[3] = self.moves_left
        # 26 elements: bitfield of all pieces
        # 26 elements: bitfield of all allied pieces
        # 26 elements: bitfield of all enemy pieces
        # 26 elements: bitfield of all allied pushers
        # 26 elements: bitfield of all enemy pushers
        color = PFPiece.White if self.white_to_move else PFPiece.Black
        arr_all = arr[4:30]
        arr_allied = arr[30:56]
        arr_enemy = arr[56:82]
        arr_allied_pushers = arr[82:108]
        arr_enemy_pushers = arr[108:134]
        for i in range(26):
            piece = self.board[i]
            if piece == PFPiece.Empty:
                continue
            arr_all[i] = 1
            if piece & color:
                arr_allied[i] = 1
                if piece == white_pusher or piece == black_pusher:
                    arr_allied_pushers[i] = 1
            else:
                arr_enemy[i] = 1
                if piece & PFPiece.Pusher:
                    arr_enemy_pushers[i] = 1
        # 26 elements: bitfield of anchor position
        if self.anchor_position > -1:
            arr[134 + self.anchor_position] = 1
        # 160 total elements
        return torch.from_numpy(arr).float().unsqueeze(0)
    
    def __repr__(self):
        formatting_tokens = ['    ', '', '\n| ', '', '', '\n| ', '', '', '', '|\n| ', '', '', '', '|\n| ', '', '', '', '|\n| ', '', '', '', '|\n    ', '', '', '|\n    ', '']
        piece_tokens = [str(p) for p in self.board]
        anchor_tokens = ['!' if i == self.anchor_position else ' ' for i in range(26)]
        board_string = ''.join(chain.from_iterable(zip(formatting_tokens, piece_tokens, anchor_tokens)))
        status_string = ' | '.join([s for s in [
            f'{"White" if self.white_to_move else "Black"} to move',
            'placing pushers' if self.num_pieces < 10 and self.num_pieces % 5 < 3 else None,
            'placing rounds' if self.num_pieces < 10 and self.num_pieces % 5 >= 3 else None,
            f'{self.moves_left} moves left' if self.num_pieces == 10 else None,
        ] if s])
        return '\n'.join([board_string, status_string, ''])
    def __str__(self): return self.__repr__()
    
    def __hash__(self):
        return hash(str(self)) # Only using this experimentally, so this doesn't need to be performant.