import random
import numpy as np
import copy 
from math import log, sqrt

from tictactoe import *

# An optimal player
class OptimalPlayer():
    def __init__(self, name = 'Optimal', verbose = False):
        self.name = name
        self.verbose = verbose

        # for a state S values[S] is the sum of the values over all plays including S
        self.values = dict()

    def play(self, state):
        '''
        returns the move to be applied from state
        '''
        if self.verbose:
            print("%s player's turn as Player %d."  % (self.name, state.player))
            print("Moves with existing values:")
            print([ (action,self.values[state.compute_new_hash(action)]) for action in state.legal_plays() \
                   if state.compute_new_hash(action) in self.values])

        # For more fun, we randomise over the most interesting moves
        if state.player == 1:
            evaluated_moves = [(self.values[state.compute_new_hash(action)], action) for action in state.legal_plays()
                               if state.compute_new_hash(action) in self.values]
            max_val, _ = max(evaluated_moves)
            interesting_moves = [action for (v,action) in evaluated_moves if v == max_val]
        else:
            evaluated_moves = [(self.values[state.compute_new_hash(action)], action) for action in state.legal_plays()
                               if state.compute_new_hash(action) in self.values]
            min_val, _ = min(evaluated_moves)
            interesting_moves = [action for (v,action) in evaluated_moves if v == min_val]
        return random.choice(interesting_moves)

    # Computes the (exact) values recursively
    def solve(self, state = State()):
        if state.compute_outcome() != -2:
            self.values[state.hash] = state.outcome
        else:
            if state.player == 1:
                current_val = -1
                for action in state.legal_plays():
                    next = state.next_state(action)
                    if not (next.hash in self.values):
                        self.solve(next)
                    current_val = max(current_val,self.values[next.hash])
                self.values[state.hash] = current_val
            else:
                current_val = 1
                for action in state.legal_plays():
                    next = state.next_state(action)
                    if not (next.hash in self.values):
                        self.solve(next)
                    current_val = min(current_val,self.values[next.hash])
                self.values[state.hash] = current_val
