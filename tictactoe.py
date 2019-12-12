import pickle
import random
import numpy as np
from math import log, sqrt

SIZE = 3
BOARD_SIZE = SIZE * SIZE

class State(object):
    def __init__(self):
        # a state is represented by a SIZE * SIZE array where:
        # 0 represents an empty position
        # 1 represents a cross (symbol for player 1),
        # 2 represents a circle (symbol for player 2)
        self.data = np.zeros((SIZE, SIZE))
        # Player who's turn it is to play from this state
        self.player = 1
        # self.outcome is the outcome of the game.
        # The value of self.outcome does not matter as long as self.over is False
        # A play won by player 1 has value 1
        # A play won by player 2 has value 0
        # A tie has value 0.5
        self.outcome = 2
        self.over = False
        self.hash = 0

    # checks whether the game is over from this state and who won
    def is_over(self):
        # checks rows
        for i in range(0, SIZE):
            if all(x == 1 for x in self.data[i, :]):
                self.outcome = 1
                self.over = True
                return True
            if all(x == 2 for x in self.data[i, :]):
                self.outcome = 0
                self.over = True
                return True

        # checks columns
        for j in range(0, SIZE):
            if all(x == 1 for x in self.data[:, j]):
                self.outcome = 1
                self.over = True
                return True
            if all(x == 2 for x in self.data[:, j]):
                self.outcome = 0
                self.over = True
                return True

        # checks diagonals
        diag = [self.data[i,i] for i in range(0, SIZE)]
        if all(x == 1 for x in diag):
            self.outcome = 1
            self.over = True
            return True
        if all(x == 2 for x in diag):
            self.outcome = 0
            self.over = True
            return True

        anti_diag = [self.data[i,SIZE - 1 - i] for i in range(0, SIZE)]
        if all(x == 1 for x in anti_diag):
            self.outcome = 1
            self.over = True
            return True
        if all(x == 2 for x in anti_diag):
            self.outcome = 0
            self.over = True
            return True

        # check whether it's a tie
        if (self.data.reshape(SIZE * SIZE) != 0).all():
            self.outcome = 0.5
            self.over = True
            return True

        # if we reached this point the game is still going on
        self.over = False

    # prints the board
    def print_state(self):
        for i in range(0, SIZE):
            print('-------------')
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
        print('-------------')

    def legal_plays(self):
        # Takes a state and returns the full list of moves that are legal moves.
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

    def next_state(self, i, j):
        # Returns the state reached after playing in (i,j)
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = self.player
        new_state.is_over()
        new_state.hash = self.hash + 3 ** (SIZE * i + j) * self.player
        new_state.player = 3 - self.player
        return new_state
     
    def update_state(self, i, j):
        # Updates the game after playing in (i,j)
        self.data[i, j] = self.player
        self.hash = self.hash + 3 ** (SIZE * i + j) * self.player
        self.player = 3 - self.player
        self.is_over()

###################################
# COMPETITION
###################################

class Competition(object):
    def save_values(self, name, player):
        # Saves the value function.
        with open('strategy_%s.bin' % name, 'wb') as f:
            pickle.dump(player.values, f)

    def load_values(self, name, player):
        # Loads a value function.
        with open('strategy_%s.bin' % name, 'rb') as f:
            player.values = pickle.load(f)

    def play(self, player1, player2, verbose=False):
        # Takes two strategies (one for each player), play them against each other once and declare a outcome
        state = State()
        state.hash = 0

        while(not state.over):
            if verbose:
                state.print_state()
            if state.player == 1:
                i, j = player1.play(state, verbose)
                state.update_state(i, j)
                if verbose:
                    print("Player %d chooses (%d,%d)" % (1, i, j))
            else:
                i, j = player2.play(state, verbose)
                state.update_state(i, j)
                if verbose:
                    print("Player %d chooses (%d,%d)" % (2, i, j))

        if verbose:
            print("Final state")
            state.print_state()
            if state.outcome == 1:
                print("Player 1 won")
            if state.outcome == 0:
                print("Player 2 won")
            if state.outcome == 0.5:
                print("It's a tie!")
        return state.outcome

    def compete(self, player1, player2, turns):
        # Takes two strategies (one for each player) and play them against each other to compute statistics
        player1_win = 0.0
        player2_win = 0.0
        for _ in range(turns):
            outcome = self.play(player1,player2)
            if outcome == 1:
                player1_win += 1
            if outcome == 0:
                player2_win += 1
        print("Results: %d plays, player 1 wins %.02f, player 2 wins %.02f" % (turns, player1_win / turns, player2_win / turns))

###################################
# GENERIC PLAYER
###################################

class Player():
    def __init__(self):
        self.values = dict()
        self.name = "Generic"

    def play(self, state, verbose = False):
        # Takes the state and returns the move to be applied.
        if not state.hash in self.values:
            if verbose:
                print("The player had never seen that state!")
            return random.choice(state.legal_plays())
        else:
            if verbose:
                print("%s player's turn as player %d.\nCurrent value: %0.2f"  % (self.name, state.player, self.values[state.hash]))
                print("Available moves and their values:")
                if state.player == 1:
                    print([((i,j),self.values[(state.next_state(i,j)).hash]) for (i,j) in state.legal_plays()
                           if (state.next_state(i,j)).hash in self.values])
                else:
                    print([((i,j),self.values[(state.next_state(i,j)).hash]) for (i,j) in state.legal_plays()
                           if (state.next_state(i,j)).hash in self.values])

            # For more fun, we randomise over the most interesting moves.
            if state.player == 1:
                max_val = max([self.values[(state.next_state(i, j)).hash] for (i, j) in state.legal_plays()
                               if (state.next_state(i, j)).hash in self.values])
                interesting_moves = [(i, j) for (i, j) in state.legal_plays() if
                                     self.values[(state.next_state(i, j)).hash] == max_val]
            else:
                min_val = min([self.values[(state.next_state(i, j)).hash] for (i, j) in state.legal_plays()
                               if (state.next_state(i, j)).hash in self.values])
                interesting_moves = [(i, j) for (i, j) in state.legal_plays() if
                                     self.values[(state.next_state(i, j)).hash] == min_val]
            return random.choice(interesting_moves)

###################################
# RANDOM PLAYER
###################################

class Player_random(Player):
    def __init__(self):
        Player.__init__(self)
        self.name = "Random"

###################################
# OPTIMAL PLAYER (MIN MAX)
###################################

class Player_optimal(Player):
    def __init__(self):
        Player.__init__(self)
        self.name = "Optimal"

    def solve(self, state = State()):
        # Computes the (exact) values recursively
        if state.over:
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

###################################
# MONTE CARLO PLAYER
###################################

class Player_MC(Player):
    def __init__(self):
        Player.__init__(self)
        self.name = "Monte Carlo"

        # For training purposes
        self.epsilon = .1

        # plays counts for each state how many plays included this state
        self.plays = dict()
        # win and loss counts for each state the number of plays won or lost.
        self.win = dict()
        self.loss = dict()

    def play_during_training(self, state):
        # Takes the state and returns the move to be applied.
        if random.random() < self.epsilon:
            return random.choice(state.legal_plays())
        else:
            possible_states = [((i, j), state.next_state(i, j)) for (i, j) in state.legal_plays()]
            if all(next.hash in self.plays for ((i, j), next) in possible_states):
                # If we have seen all of the legal moves at least once, we use the UCB bound.
                if state.player == 1:
                    _, (i, j) = max(((self.win[next.hash] - self.loss[next.hash]) / self.plays[next.hash], (i, j)) for ((i, j), next) in possible_states)
                else:
                    _, (i, j) = min(((self.win[next.hash] - self.loss[next.hash]) / self.plays[next.hash], (i, j)) for ((i, j), next) in possible_states)
            else:
                # Otherwise choose randomly among unevaluated moves
                unevaluated_moves = [(i, j) for (i, j) in state.legal_plays() if
                                     not (state.next_state(i, j)).hash in self.plays]
                (i, j) = random.choice(unevaluated_moves)
            return i,j

    def store_new_state(self, state):
        if not(state.hash in self.plays):
            self.plays[state.hash] = 0
            self.win[state.hash] = 0
            self.loss[state.hash] = 0
            self.values[state.hash] = 0

    def run_simulation(self):
        state = State()
        state.hash = 0
        self.store_new_state(state)
        self.plays[state.hash] += 1

        # We store the play in a sequence
        play = []

        while not state.over:
            play.append(state.hash)
            i, j = self.play_during_training(state)
            state.update_state(i, j)
            self.store_new_state(state)
            self.plays[state.hash] += 1

        if state.outcome == 1:
            for hash_val in play:
                self.win[hash_val] += 1
        if state.outcome == 0:
            for hash_val in play:
                self.loss[hash_val] += 1

    def train(self, number_simulations, verbose = False):
        # Approximates the values through Monte Carlo simulation.
        t = 1
        while t <= number_simulations:
            self.run_simulation()
            if verbose and t % 100 == 0:
                print("number of plays %d, number of wins %d, number of losses %d" % (self.plays[0], self.win[0], self.loss[0]))
                print("value: %0.2f" % (1/2 * (self.plays[0] + self.win[0] - self.loss[0]) / self.plays[0]))
            t += 1

        # Update the values.
        for hash_val in self.plays:
            self.values[hash_val] = 1/2 * (self.plays[hash_val] + self.win[hash_val] - self.loss[hash_val]) / self.plays[hash_val]


###################################
# UCB PLAYER
###################################

class Player_UCB(Player):
    def __init__(self):
        Player.__init__(self)
        self.name = "UCB"

        # For training purposes
        self.C = 1.4

        # plays counts for each state how many plays included this state
        self.plays = dict()
        # win and loss counts for each state the number of plays won or lost.
        self.win = dict()
        self.loss = dict()

    def play_during_training(self, state):
        # Takes the state and returns the move to be applied.
        possible_states = [((i, j), state.next_state(i, j)) for (i, j) in state.legal_plays()]
        if all(next.hash in self.plays for ((i, j), next) in possible_states):
            # If we have seen all of the legal moves at least once, we use the UCB bound.
            if state.player == 1:
                _, (i, j) = max(
                    (self.win[next.hash] / self.plays[next.hash] +
                     self.C * sqrt(log(self.plays[state.hash]) / self.plays[next.hash]), (i, j))
                    for ((i, j), next) in possible_states)
            else:
                _, (i, j) = min(
                    (self.win[next.hash] / self.plays[next.hash] +
                     self.C * sqrt(log(self.plays[state.hash]) / self.plays[next.hash]), (i, j))
                    for ((i, j), next) in possible_states)
        else:
            # Otherwise choose randomly among unevaluated moves
            unevaluated_moves = [(i, j) for (i, j) in state.legal_plays() if
                                 not (state.next_state(i, j)).hash in self.plays]
            (i, j) = random.choice(unevaluated_moves)
        return i,j

    def store_new_state(self, state):
        if not(state.hash in self.plays):
            self.plays[state.hash] = 0
            self.win[state.hash] = 0
            self.loss[state.hash] = 0
            self.values[state.hash] = 0

    def run_simulation(self):
        state = State()
        state.hash = 0
        self.store_new_state(state)
        self.plays[state.hash] += 1

        # We store the play in a sequence
        play = []

        while not state.over:
            play.append(state.hash)
            i, j = self.play_during_training(state)
            state.update_state(i, j)
            self.store_new_state(state)
            self.plays[state.hash] += 1

        if state.outcome == 1:
            for hash_val in play:
                self.win[hash_val] += 1
        if state.outcome == 0:
            for hash_val in play:
                self.loss[hash_val] += 1

    def train(self, number_simulations, verbose = False):
        # Approximates the values through Monte Carlo simulation.
        t = 1
        while t <= number_simulations:
            self.run_simulation()
            if verbose and t % 100 == 0:
                print("Number of plays %d, number of wins %d, number of losses %d" % (self.plays[0], self.win[0], self.loss[0]))
                print("Value: %0.2f" % (1/2 * (self.plays[0] + self.win[0] - self.loss[0]) / self.plays[0]))
            t += 1

        # Update the values.
        for hash_val in self.plays:
            self.values[hash_val] = 1/2 * (self.plays[hash_val] + self.win[hash_val] - self.loss[hash_val]) / self.plays[hash_val]

###################################
# TD PLAYER
###################################

class Player_TD(Player):
    def __init__(self):
        Player.__init__(self)
        self.name = "TD"

        # For training purposes
        self.step_size = 0.1
        self.epsilon = 0.1

    def play_during_training(self, state):
        # Takes the state and returns the move to be applied.
        # The boolean says whether the move was random (False) or greedy (True)
        if random.random() < self.epsilon:
            return (False,random.choice(state.legal_plays()))
        else:
            possible_states = [((i, j), state.next_state(i, j)) for (i, j) in state.legal_plays()]
            if all(next.hash in self.values for ((i, j), next) in possible_states):
                # If we have seen all of the legal moves at least once, we use the value.
                if state.player == 1:
                    _, (i, j) = max((self.values[next.hash], (i, j)) for ((i, j), next) in possible_states)
                else:
                    _, (i, j) = min((self.values[next.hash], (i, j)) for ((i, j), next) in possible_states)
            else:
                # Otherwise choose randomly among unevaluated moves
                unevaluated_moves = [(i, j) for (i, j) in state.legal_plays() if
                                     not (state.next_state(i, j)).hash in self.values]
                (i, j) = random.choice(unevaluated_moves)
            return (True,(i, j))

    def store_new_state(self, state):
        if not (state.hash in self.values):
            self.values[state.hash] = 0

    def run_simulation(self):
        state = State()
        state.hash = 0
        self.store_new_state(state)

        play = []
        while not state.over:
            greedy, (i, j) = self.play_during_training(state)
            play.append((state.hash,greedy))
            state.update_state(i, j)
            new_hash_val = state.hash
            self.store_new_state(state)

        next_hash_val,_ = play[-1]
        self.values[next_hash_val] = state.outcome
        for (hash_val,greedy) in reversed(play[:-1]):
            if greedy:
                td_error = self.values[next_hash_val] - self.values[hash_val]
                self.values[hash_val] += self.step_size * td_error

    def train(self, number_simulations, verbose=False):
        # Approximates the values through Temporal Difference.
        t = 1
        while t <= number_simulations:
            self.run_simulation()
            if verbose and t % 100 == 0:
                print("Value: %0.2f" % self.values[0])
            t += 1

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

player_rand = Player_random()

player_optimal = Player_optimal()
#player_optimal.solve()
#competition.save_values("optimal", player_optimal)
competition.load_values("optimal", player_optimal)

player_mc = Player_MC()
#player_mc.train(20000, True)
#competition.save_values("MC_20000", player_mc)
competition.load_values("MC_20000", player_mc)

player_ucb = Player_UCB()
#player_ucb.train(10000, True)
#competition.save_values("UCB_10000", player_ucb)
competition.load_values("UCB_10000", player_ucb)

player_td = Player_TD()
#player_td.train(2000, True)
#competition.save_values("TD_2000", player_td)
competition.load_values("TD_2000", player_td)

competition.play(player_td, player_optimal, verbose=True)
competition.compete(player_td, player_optimal, 500)



#### How accurate is Player_mc's value function?
#_, hash_val = max((abs(player_optimal.values[hash_val] - player_mc.values[hash_val]), hash_val) for hash_val in player_mc.plays)
#print("How bad is player_mc? Here is the state where they disagree the most:")
#State.print_state(unhash(hash_val))
#print("The optimal value is %0.2f, the value according to player_mc is %0.2f, over a sample of %d plays" %
#(player_optimal.values[hash_val], player_mc.values[hash_val], player_mc.plays[hash_val]))

##### Comparing player_mc together
#player_mc1 = Player_MC()
#player_mc2 = Player_MC()

#player_mc1.train(1000)
#player_mc2.train(20000)
#competition.save_values("MC_1000", player_mc1)
#competition.save_values("MC_20000", player_mc2)

#competition.load_values("MC_5000", player_mc1)
#competition.load_values("MC_20000", player_mc2)

#competition.play(player_mc1, player_mc2, verbose=True)
#competition.compete(player_mc1, player_mc2, 500)

#print(len(player_optimal.values))
