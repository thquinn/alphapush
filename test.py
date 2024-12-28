import random
from pushfight import PFPiece, PFState

state = PFState()
while (state.winner == PFPiece.Empty):
    state.move(random.choice(state.moves))
    print(state)
    print()