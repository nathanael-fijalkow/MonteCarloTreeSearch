import pickle
import numpy as np
from tqdm import tqdm

from tictactoe import *
from MCTS import *
from optimal import *

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

        count_opt = 0
        play_length = 0
        if verbose:
            print("\nMatch between Player %s (as Player 1) and Player %s (as Player 2)"
                  % (player1.name, player2.name))

        while state.compute_outcome() == -2:
            if verbose:
                print(state)
            if state.player == 1:
                action = player1.play(state)
                if player_of_interest == 1:
                    is_optimal = player2.values[state.hash] <= player2.values[state.compute_new_hash(action)]
                    count_opt += is_optimal
                    play_length += 1
                if verbose:
                    print("Player %d chooses (%d,%d)" % (1, action[0], action[1]))
                    if player_of_interest == 1 and is_optimal:
                        print("This was an optimal move (the current value is %0.1f)" %
                              player2.values[state.compute_new_hash(action)])
                state.update_state(action)
            else:
                action = player2.play(state)
                if player_of_interest == 2:
                    is_optimal = player1.values[state.hash] >= player1.values[state.compute_new_hash(action)]
                    count_opt += is_optimal
                    play_length += 1
                if verbose:
                    print("Player %d chooses (%d,%d)" % (2, action[0], action[1]))
                    if player_of_interest == 2 and is_optimal:
                        print("This was an optimal move (the current value is %0.1f)" %
                              player1.values[state.compute_new_hash(action)])
                state.update_state(action)

        if verbose:
            print("Final state")
            print(state)
            if state.outcome == 1:
                print("Player 1 won")
            elif state.outcome == -1:
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
            if outcome == -1:
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
            # If Player 1 loses the outcome is -1
            # If Player 2 loses the outcome is 1
            outcome,_,_ = self.play(player1,player2, verbose=False)
            if player_of_interest == 1 and outcome == -1:
                return i
            if player_of_interest == 2 and outcome == 1:
                return i
            i += 1
        return games

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

player_optimal = OptimalPlayer(name = "Optimal")
player_optimal.solve()

# competition.save_values("optimal", player_optimal)
# competition.load_values("optimal", player_optimal)

player_eps_average = MCTSPlayer(name='epsilon-greedy average sample', strategy='epsilon-greedy', update_mode='average')
player_eps_step_size = MCTSPlayer(name='epsilon-greedy step size', strategy='epsilon-greedy', update_mode='step_size')
player_eps_td = MCTSPlayer(name='epsilon-greedy TD', strategy='epsilon-greedy', update_mode='TD', step_size=0.5)

player_UCB_average = MCTSPlayer(name='UCB average sample', strategy='UCB', update_mode='average')
player_UCB_step_size = MCTSPlayer(name='UCB step size', strategy='UCB', update_mode='step_size')
player_UCB_td = MCTSPlayer(name='UCB TD', verbose = False, strategy='UCB', update_mode='TD', step_size=0.5)



# competition.play(player_optimal,player_optimal,verbose=True)

competition.play(player_UCB_td, player_optimal, verbose=True)
# player_UCB_td.self_play()

#competition.compete(player_eps_td, player_optimal, player_of_interest=1)
#competition.compete(player_optimal, player_eps_td, player_of_interest=2)

################################################
# How many iterations to ensure a tie against the optimal player?
################################################

def how_many_iterations(player,steps = 100, games = 500, verbose = False, player_of_interest = 1):
    if verbose:
        print("\nPlayer %s: how many iterations to win when playing %s" % (player.name, "first" if player_of_interest == 1 else "second"))
    iteration = 0
    while(True):
        player.self_play(steps)
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

# how_many_iterations(player_eps_average, steps = 1000, games = 100, verbose = True, player_of_interest = 2)
# how_many_iterations(player_eps_step_size, steps = 1000, games = 100, verbose = True, player_of_interest = 2)
# how_many_iterations(player_eps_td, steps = 1000, games = 100, verbose = True, player_of_interest = 1)

# how_many_iterations(player_UCB_average, steps = 1000, games = 100, verbose = True, player_of_interest = 1)
# how_many_iterations(player_UCB_step_size, steps = 1000, games = 100, verbose = True, player_of_interest = 1)
# how_many_iterations(player_UCB_td, steps = 1000, games = 100, verbose = True, player_of_interest = 2)

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
                player = MCTSPlayer(strategy=strategy, update_mode=update_mode, epsilon=parameter)
            if parameter_name == 'UCB':
                player = MCTSPlayer(strategy=strategy, update_mode=update_mode, UCB=parameter)
            if parameter_name == 'step size':
                player = MCTSPlayer(strategy=strategy, update_mode=update_mode, step_size=parameter)
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

