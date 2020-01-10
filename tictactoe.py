import pickle
import random
import numpy as np
from math import log, sqrt
from tqdm import tqdm

SIZE = 3

class State(object):
    def __init__(self):
        # @data is a SIZE * SIZE array where
        # 0 represents an empty position
        # 1 represents a cross (symbol for player 1)
        # 2 represents a circle (symbol for player 2)
        self.data = np.zeros((SIZE, SIZE))
        # @player: who's turn it is to play from this state
        self.player = 1
        self.hash = 0
        # @outcome can be
        # 1 if Player 1 wins
        # 0 if Player 2 wins
        # 0.5 if it's a tie
        # -1 if the game is not over
        # 2 if the outcome has never been computed
        self.outcome = 2

    # Checks whether the game is over from this state and who won
    def compute_outcome(self):
        if self.outcome != 2:
            return self.outcome
        else:
            # Checks rows
            for i in range(0, SIZE):
                if all(x == 1 for x in self.data[i, :]):
                    self.outcome = 1
                    return 1
                if all(x == 2 for x in self.data[i, :]):
                    self.outcome = 0
                    return 0

            # Checks columns
            for j in range(0, SIZE):
                if all(x == 1 for x in self.data[:, j]):
                    self.outcome = 1
                    return 1
                if all(x == 2 for x in self.data[:, j]):
                    self.outcome = 0
                    return 0

            # Checks diagonals
            diag = [self.data[i,i] for i in range(0, SIZE)]
            if all(x == 1 for x in diag):
                self.outcome = 1
                return 1
            if all(x == 2 for x in diag):
                self.outcome = 0
                return 0

            anti_diag = [self.data[i,SIZE - 1 - i] for i in range(0, SIZE)]
            if all(x == 1 for x in anti_diag):
                self.outcome = 1
                return 1
            if all(x == 2 for x in anti_diag):
                self.outcome = 0
                return 0

            # Checks whether it's a tie
            data_all = [self.data[i,j] for i in range(0, SIZE) for j in range(0, SIZE)]
            if all(x != 0 for x in data_all):
                self.outcome = 0.5
                return 0.5

            # If we reached this point the game is still going on
            self.outcome = -1
            return -1

    # Prints the board
    def print_state(self):
        for i in range(0, SIZE):
            out = '-'
            for _ in range(0, SIZE):
                out += '----'
            print(out)
            out = '| '
            for j in range(0, SIZE):
                if self.data[i, j] == 1:
                    token = 'x'
                elif self.data[i, j] == 2:
                    token = 'o'
                else:
                    token = ' '
                out += token + ' | '
            print(out)
        out = ''
        for _ in range(0, SIZE):
            out += '----'
        print(out)

    # Takes a state and returns the full list of moves that are legal moves
    def legal_plays(self):
        legal = []
        for i in range(0, SIZE):
            for j in range(0, SIZE):
                if self.data[i, j] == 0:
                    legal.append((i,j))
        return legal

    # Actually not useful because hashes are computed recursively
    def compute_hash(self):
        self.hash = 0
        for i in self.data.reshape(SIZE * SIZE):
            self.hash = self.hash * 3 + i
        return self.hash

    # Compute the hash of the state obtained by playing in (i,j)
    def update_hash(self, i, j):
        return self.hash + 3 ** (SIZE * i + j) * self.player

    # Returns a new state obtained by playing in (i,j)
    def next_state(self, i, j):
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = self.player
        new_state.hash = self.update_hash(i,j)
        new_state.player = 3 - self.player
        return new_state

    # Updates the same state by playing in (i,j)
    def update_state(self, i, j):
        self.data[i, j] = self.player
        self.hash = self.update_hash(i,j)
        self.player = 3 - self.player
        self.outcome = 2

###################################
# COMPETITION
###################################

class Competition(object):
    # Saves the value function
    def save_values(self, name, player):
        with open('strategy_%s.bin' % name, 'wb') as f:
            pickle.dump(player.values, f)

    # Loads a value function
    def load_values(self, name, player):
        with open('strategy_%s.bin' % name, 'rb') as f:
            player.values = pickle.load(f)

    # Takes two strategies (one for each player), play them against each other once and declare an outcome
    # if player_of_interest is 1 or 2, it assumes that the other player is optimal
    def play(self, player1, player2, verbose=False, player_of_interest = 0):
        state = State()
        state.hash = 0

        count_opt = 0
        play_length = 0
        if verbose:
            print("\nMatch between Player %s (as Player 1) and Player %s (as Player 2)"
                  % (player1.name, player2.name))

        while state.compute_outcome() == -1:
            if verbose:
                state.print_state()
            if state.player == 1:
                i, j = player1.play(state, verbose)
                if player_of_interest == 1:
                    is_optimal = player2.values[state.hash] <= player2.values[state.update_hash(i, j)]
                    count_opt += is_optimal
                    play_length += 1
                if verbose:
                    print("Player %d chooses (%d,%d)" % (1, i, j))
                    if player_of_interest == 1 and is_optimal:
                        print("This was an optimal move, the current value is %0.1f" %
                              player2.values[state.update_hash(i, j)])
                state.update_state(i, j)
            else:
                i, j = player2.play(state, verbose)
                if player_of_interest == 2:
                    is_optimal = player1.values[state.hash] >= player1.values[state.update_hash(i, j)]
                    count_opt += is_optimal
                    play_length += 1
                if verbose:
                    print("Player %d chooses (%d,%d)" % (2, i, j))
                    if player_of_interest == 2 and is_optimal:
                        print("This was an optimal move (the current value is %0.1f)" %
                              player1.values[state.update_hash(i, j)])
                state.update_state(i, j)

        if verbose:
            print("Final state")
            state.print_state()
            if state.outcome == 1:
                print("Player 1 won")
            elif state.outcome == 0:
                print("Player 2 won")
            else:
                print("It's a tie!")
        return state.outcome,count_opt,play_length

    # Takes two strategies (one for each player) and play them against each other for a number of games
    # if player_of_interest is 1 or 2, it assumes that the other player is optimal
    def compete(self, player1, player2, games = 500, player_of_interest = 0):
        player1_win = 0.0
        player2_win = 0.0
        count_opt_tot = 0
        count_length_tot = 0
        for _ in range(games):
            outcome,count_opt,play_length = \
                self.play(player1,player2, verbose = False, player_of_interest = player_of_interest)
            count_opt_tot += count_opt
            count_length_tot += play_length
            if outcome == 1:
                player1_win += 1
            if outcome == 0:
                player2_win += 1
        print("\nCompetition of Player %s (as Player 1) against Player %s (as Player 2):"
              "\n %d plays, Player 1 wins %.02f, Player 2 wins %.02f"
              % (player1.name, player2.name, games, player1_win / games, player2_win / games))
        if player_of_interest:
            print("Player %s played optimal moves %0.2f percent of the time" %
                  (player1.name if player_of_interest == 1 else player2.name,
                   count_opt_tot / count_length_tot * 100))

    # Checks whether a player ensures ties against another player over a number of games
    def ensures_tie(self, player1, player2, games = 50, player_of_interest = 1):
        i = 0
        while i < games:
            # If the player of interest loses, stop
            # Reminder:
            # If Player 1 loses the outcome is 0
            # If Player 2 loses the outcome is 1
            outcome,_,_ = self.play(player1,player2, verbose=False)
            if outcome == player_of_interest - 1:
                return i
            i += 1
        return games

###################################
# PLAYER
###################################

class Player():
    def __init__(self, name = 'Anonymous', strategy ='epsilon-greedy', update_mode = 'average',
                 epsilon = 0.2, UCB = 1.5, step_size = 0.1):
        self.values = dict()
        self.name = name
        # @plays counts for each state how many plays included this state
        self.plays = dict()

        # What strategy are we using during training: 'epsilon-greedy' or 'UCB'
        self.strategy = strategy

        # How do we update estimates: 'average' or 'step_size' or 'TD'
        self.update_mode = update_mode

        # Parameters
        self.epsilon = epsilon
        self.UCB = UCB
        self.step_size = step_size

    # Takes the state and returns the move to be applied
    def play(self, state, verbose = False):
        if not state.hash in self.values:
            if verbose:
                print("The player had never seen that state!")
            return random.choice(state.legal_plays())
        else:
            if verbose:
                print("%s player's turn as Player %d.\nCurrent value: %0.5f"  % (self.name, state.player, self.values[state.hash]))
                print("Available moves and their values:")
                print([((i,j),self.values[state.update_hash(i,j)]) for (i,j) in state.legal_plays()
                       if state.update_hash(i,j) in self.values])

            # For more fun, we randomise over the most interesting moves
            if state.player == 1:
                evaluated_moves = [(self.values[state.update_hash(i,j)], (i,j)) for (i, j) in state.legal_plays()
                                   if state.update_hash(i,j) in self.values]
                max_val, _ = max(evaluated_moves)
                interesting_moves = [(i, j) for (v,(i, j)) in evaluated_moves if v == max_val]
            else:
                evaluated_moves = [(self.values[state.update_hash(i,j)], (i,j)) for (i, j) in state.legal_plays()
                                   if state.update_hash(i,j) in self.values]
                min_val, _ = min(evaluated_moves)
                interesting_moves = [(i, j) for (v,(i, j)) in evaluated_moves if v == min_val]
            return random.choice(interesting_moves)

    # Computes the (exact) values recursively
    def solve(self, state = State()):
        print(len(self.values))
        if state.compute_outcome() != -1:
            self.values[state.hash] = state.outcome
        else:
            if state.player == 1:
                current_val = 0
                for (i,j) in state.legal_plays():
                    next = state.next_state(i,j)
                    if not (next.hash in self.values):
                        self.solve(next)
                    current_val = max(current_val,self.values[next.hash])
                self.values[state.hash] = current_val
            else:
                current_val = 1
                for (i,j) in state.legal_plays():
                    next = state.next_state(i,j)
                    if not (next.hash in self.values):
                        self.solve(next)
                    current_val = min(current_val,self.values[next.hash])
                self.values[state.hash] = current_val

    # During training, takes the current state and returns the move to be applied
    # The boolean says whether the move was chosen greedily (True) or uniformly at random (False)
    def play_during_training(self, state, t):
        possible_states = [((i, j), state.update_hash(i, j)) for (i, j) in state.legal_plays()]
        # If we have seen all of the legal moves at least once
        if all(hash_val in self.plays for ((i, j), hash_val) in possible_states):
            if self.strategy == 'epsilon-greedy':
            # Play the epsilon-greedy strategy
                if random.random() < self.epsilon:
                    return random.choice(state.legal_plays())
                else:
                    if state.player == 1:
                        _, (i, j) = max((self.values[hash_val], (i, j)) for ((i, j), hash_val) in possible_states)
                    else:
                        _, (i, j) = min((self.values[hash_val], (i, j)) for ((i, j), hash_val) in possible_states)

            if self.strategy == 'UCB':
            # Play the UCB strategy
                if state.player == 1:
                    _, (i, j) = max(
                        (self.values[hash_val] +
                         self.UCB * sqrt(log(self.plays[state.hash]) / self.plays[hash_val]), (i, j))
                        for ((i, j), hash_val) in possible_states)
                else:
                    _, (i, j) = min(
                        (self.values[hash_val] -
                         self.UCB * sqrt(log(self.plays[state.hash]) / self.plays[hash_val]), (i, j))
                        for ((i, j), hash_val) in possible_states)
            return i, j
        else:
        # Otherwise choose randomly among unevaluated moves
            unevaluated_moves = [(i, j) for (i, j) in state.legal_plays() if
                                 not state.update_hash(i,j) in self.plays]
            (i, j) = random.choice(unevaluated_moves)
            return i, j

    def store_new_state(self, state):
        if not(state.hash in self.plays):
            self.plays[state.hash] = 0
            self.values[state.hash] = 0.5

    def run_simulation(self, t):
        state = State()
        state.hash = 0
        self.store_new_state(state)
        self.plays[state.hash] += 1

        # We store the play in a sequence
        play = []

        while state.compute_outcome() == -1:
            (i, j) = self.play_during_training(state,t)
            play.append(state.hash)
            state.update_state(i, j)
            self.store_new_state(state)
            self.plays[state.hash] += 1

        if self.update_mode == 'average':
            # Update using average
            self.values[state.hash] = state.outcome
            for hash_val in play:
                self.values[hash_val] += 1.0 / self.plays[state.hash] * (state.outcome - self.values[hash_val])

        if self.update_mode == 'step_size':
            # Update using step size
            self.values[state.hash] = state.outcome
            for hash_val in play:
                self.values[hash_val] += self.step_size * (state.outcome - self.values[hash_val])

        if self.update_mode == 'TD':
            # Update using temporal difference (TD)
            next_hash_val = state.hash
            self.values[next_hash_val] = state.outcome
            for hash_val in reversed(play):
                td_error = self.values[next_hash_val] - self.values[hash_val]
                self.values[hash_val] += self.step_size * td_error
                next_hash_val = hash_val

    def train(self, number_simulations, verbose = False, steps = 100):
        if verbose:
            print("\nStart training of Player %s" % self.name)
        for t in range(1,number_simulations+1):
            self.run_simulation(t)
            if verbose and t % steps == 0:
                print("After %d iterations the value of the initial state is %0.5f" % (t, self.values[0]))

###################################
# TESTS
###################################

def unhash(hash_val):
    state = State()
    for i in range(SIZE * SIZE):
        state.data[i % SIZE, int(i / SIZE)] = hash_val % SIZE
        hash_val = int(hash_val / SIZE)
    return state

competition = Competition()

player_optimal = Player(name = "Optimal")
#player_optimal.solve()
#competition.save_values("optimal", player_optimal)
competition.load_values("optimal", player_optimal)

player_eps_average = Player(name='epsilon-greedy average sample', strategy='epsilon-greedy', update_mode='average')
player_eps_step_size = Player(name='epsilon-greedy step size', strategy='epsilon-greedy', update_mode='step_size')
player_eps_td = Player(name='epsilon-greedy TD', strategy='epsilon-greedy', update_mode='TD', step_size=0.5)

player_UCB_average = Player(name='UCB average sample', strategy='UCB', update_mode='average')
player_UCB_step_size = Player(name='UCB step size', strategy='UCB', update_mode='step_size')
player_UCB_td = Player(name='UCB TD', strategy='UCB', update_mode='TD', step_size=0.5)

# Test player_eps_td
#player_eps_td.train(10000, True, steps = 1000)
#competition.compete(player_eps_td, player_optimal, player_of_interest=1)
#competition.compete(player_optimal, player_eps_td, player_of_interest=2)
#competition.play(player_eps_td,player_optimal,verbose=True)

# Play player_eps_td against player_UCB_td
#player_eps_td.train(10000, False)
#player_UCB_td.train(10000, False)
#competition.compete(player_eps_td,player_UCB_td)
#competition.compete(player_UCB_td, player_eps_td)
#competition.play(player_eps_td,player_UCB_td,verbose=True)

################################################
# Print the number of reachable states
################################################

#print("Number of reachable states: %d" % len(player_optimal.values))

################################################
# How many iterations to ensure a tie against the optimal player?
################################################

def how_many_iterations(player,steps = 100, games = 500, verbose = False, player_of_interest = 1):
    if verbose:
        print("\nPlayer %s: how many iterations to win when playing %s" % (player.name, "first" if player_of_interest == 1 else "second"))
    iteration = 0
    while(True):
        player.train(steps)
        iteration += steps
        if player_of_interest == 1:
            result = competition.ensures_tie(player,player_optimal,games,player_of_interest)
        if player_of_interest == 2:
            result = competition.ensures_tie(player_optimal,player,games,player_of_interest)
        if result == games:
            if verbose:
                print("Over: after %d iterations, Player %s ensured ties each of the %d matches"
                      % (iteration, player.name,games))
            return iteration
        elif verbose:
            print("After %d iterations, Player %s lost the match number %d" % (iteration, player.name, result))

#how_many_iterations(player_eps_average, steps = 1000, games = 100, verbose = True, player_of_interest = 2)
#how_many_iterations(player_eps_step_size, steps = 1000, games = 100, verbose = True, player_of_interest = 2)
#how_many_iterations(player_eps_td, steps = 1000, games = 100, verbose = True, player_of_interest = 1)

#how_many_iterations(player_UCB_average, steps = 1000, games = 100, verbose = True, player_of_interest = 1)
#how_many_iterations(player_UCB_step_size, steps = 1000, games = 100, verbose = True, player_of_interest = 1)
#how_many_iterations(player_UCB_td, steps = 1000, games = 100, verbose = True, player_of_interest = 2)

################################################
# Parameter tuning 
################################################

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def statistics(strategy, update_mode, parameter_name, parameter_list, number_tests, steps, games):
    out = [[] for i in range(len(parameter_list))]
    print("Statistics for the parameter %s" % parameter_name)
    for index,parameter in enumerate(parameter_list):
        print("parameter: %0.2f" % parameter)
        for i in tqdm(range(number_tests)):
            if parameter_name == 'epsilon':
                player = Player(strategy=strategy, update_mode=update_mode, epsilon=parameter)
            if parameter_name == 'UCB':
                player = Player(strategy=strategy, update_mode=update_mode, UCB=parameter)
            if parameter_name == 'step size':
                player = Player(strategy=strategy, update_mode=update_mode, step_size=parameter)
            out[index].append(how_many_iterations(player,steps,games,verbose=False))
    with open('statistics_%s_%s_%s.bin' % (strategy, update_mode, parameter_name), 'wb') as f:
        pickle.dump(out, f)
    fig, ax = plt.subplots()
    ax.violinplot(out,parameter_list,widths=0.03)
    ax.set_xlabel("Parameter: %s" %parameter_name)
    ax.set_ylabel("Number of iterations")
    plt.savefig('statistics_%s_%s_%s.png' % (strategy, update_mode, parameter_name))
    plt.close()

parameter_list_eps_td = np.arange(0.05,0.55,step = 0.05)
#statistics(strategy='epsilon-greedy', update_mode='TD', parameter_name = 'epsilon', parameter_list = parameter_list_eps_td, number_tests = 50, steps = 250, games = 50)

parameter_list_ucb_td = [1 + (i + 1) / 10 for i in range(10)]
#statistics(strategy='UCB', update_mode='TD', parameter_name = 'UCB', parameter_list = parameter_list_ucb_td, number_tests = 50, steps = 250, games = 50)

parameter_list_eps_step_size = np.arange(0.05,0.3,step = 0.05)
#statistics(strategy='epsilon-greedy', update_mode='TD', parameter_name = 'step size', parameter_list = parameter_list_eps_step_size, number_tests = 50, steps = 250, games = 50)

