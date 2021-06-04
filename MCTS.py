import random
import numpy as np
import copy 
from math import log, sqrt

from tictactoe import *

class MCTSPlayer():
    def __init__(self, 
        name = 'Anonymous',
        verbose = False, 
        strategy ='epsilon-greedy', 
        update_mode = 'average',
        epsilon = 0.2, 
        UCB = 1.5, 
        step_size = 0.1,
        nb_simulations = 500):
        self.name = name
        self.verbose = verbose

        # STATISTICS
        # for a state S values[S] is the sum of the values over all plays including S
        self.values = dict()
        # for a state S plays[S] is the number of plays including S
        self.plays = dict()

        # self.history will be used for simulations
        self.history = []

        # ALGORITHM CHOICES
        # what strategy are we using during selection: 'epsilon-greedy' or 'UCB'
        self.strategy = strategy
        # how do we update estimates: 'average' or 'step_size' or 'TD'
        self.update_mode = update_mode

        # ALGORITHM PARAMETERS
        self.epsilon = epsilon
        self.UCB = UCB
        self.step_size = step_size
        self.nb_simulations = nb_simulations

    def play(self, state):
        '''
        returns the move to be applied from state.
        To do this we run a number of simulations from state (updating the statistics)
        and pick the action having the best statistics
        '''
        legal_plays = state.legal_plays()
        if len(legal_plays) == 1:
            return legal_plays[0]

        if state.hash not in self.values:
            self.plays[state.hash] = 0
            self.values[state.hash] = 0

        for i in range(self.nb_simulations):
            if self.verbose:
                print("Start simulation {} from {}".format(i, state))

            self.history.clear()
            new_state = copy.deepcopy(state)
            self.history.append(new_state.hash)

            # SELECTION: 
            # picks a successor by applying the strategy (epsilon-greedy or UCB)
            # as long as statistics exist for all successors

            possible_states = [(action, new_state.compute_new_hash(action)) for action in new_state.legal_plays()]
            while new_state.compute_outcome() == -2 \
            and all(hash_val in self.plays for (action, hash_val) in possible_states):
                action = self.selection(new_state, possible_states)
                new_state.update_state(action)
                self.history.append(new_state.hash)
                possible_states = [(action, new_state.compute_new_hash(action)) for action in new_state.legal_plays()]
                if self.verbose:
                    print("In selection phase added action {} leading to {}".format(action, new_state))

            # EXPANSION:
            # picks a random successor (with no statistics) and adds it to the statistics

            if new_state.compute_outcome() == -2:
                legal_not_in_values = [action for action in new_state.legal_plays() \
                if new_state.compute_new_hash(action) not in self.values]
                action = random.choice(legal_not_in_values)
                new_state.update_state(action)
                if self.verbose:
                    print("In expansion phase added action {} leading to {}".format(action, new_state))
                self.history.append(new_state.hash)
                self.plays[new_state.hash] = 0
                self.values[new_state.hash] = 0

            # (LIGHT) PLAYOUT:
            # picks random successors until the end of the play

            while new_state.compute_outcome() == -2:
                action = random.choice(new_state.legal_plays())
                new_state.update_state(action)
                if self.verbose:
                    print("In playout phase added action {} leading to {}".format(action, new_state))

            # BACKPROPAGATION:
            # updates the statistics of states in history
            outcome = new_state.compute_outcome()
            if self.verbose:
                print("Entering backpropagation with history \n{} and outcome {}".format(self.history, outcome))
            self.backpropagation(outcome)

        # We can now trust the statistics
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
        if self.verbose:
            print("Set of interesting_moves:\n{}".format(interesting_moves))
        return random.choice(interesting_moves)

    def selection(self, state, possible_states):
        '''
        pick a successor by applying the strategy (epsilon-greedy or UCB)
        '''
        if self.strategy == 'epsilon-greedy':
        # Play the epsilon-greedy strategy
            if random.random() < self.epsilon:
                action = random.choice(state.legal_plays())
            else:
                if state.player == 1:
                    _, action = max((self.values[hash_val], action) for (action, hash_val) in possible_states)
                else:
                    _, action = min((self.values[hash_val], action) for (action, hash_val) in possible_states)

        if self.strategy == 'UCB':
        # Play the UCB strategy
            if state.player == 1:
                _, action = max(
                    (self.values[hash_val] +
                     self.UCB * sqrt(log(self.plays[state.hash]) / self.plays[hash_val]), action)
                    for (action, hash_val) in possible_states)
            else:
                _, action = min(
                    (self.values[hash_val] -
                     self.UCB * sqrt(log(self.plays[state.hash]) / self.plays[hash_val]), action)
                    for (action, hash_val) in possible_states)
        return action

    def backpropagation(self, outcome):
        '''
        backpropagate the values over history
        '''
        last_state_hash = self.history.pop()
        self.plays[last_state_hash] += 1
        self.values[last_state_hash] = outcome

        if self.update_mode == 'average':
            # Update using average
            for hash_val in self.history:
                self.plays[hash_val] += 1
                self.values[hash_val] += 1 / self.plays[hash_val] * (outcome - self.values[hash_val])

        if self.update_mode == 'step_size':
            # Update using step size
            for hash_val in self.history:
                self.plays[hash_val] += 1
                self.values[hash_val] += self.step_size * (outcome - self.values[hash_val])

        if self.update_mode == 'TD':
            # Update using temporal difference (TD)
            next_hash_val = last_state_hash
            for hash_val in reversed(self.history):
                self.plays[hash_val] += 1
                td_error = self.values[next_hash_val] - self.values[hash_val]
                self.values[hash_val] += self.step_size * td_error
                next_hash_val = hash_val

    def self_play(self, number_training_plays = 50, steps = 5):
        if self.verbose:
            print("\nStart training of Player %s" % self.name)
        for t in range(1,number_training_plays+1):
            state = State()
            while state.compute_outcome() == -2:
                action = self.play(state)
                state.update_state(action)
            if t % steps == 0:
                print("After %d iterations the value of the initial state is %0.5f" % (t, self.values[0]))
