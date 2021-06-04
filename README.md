# MonteCarloTreeSearch
This repository contains a clean, simple, and generic implementation of Monte Carlo Tree Search. 

For testing purposes it is applied to the game of Tic Tac Toe through a simple API
which is (heavily) inspired by Jeff Bradberry's own code (https://github.com/jbradberry).

### Contents
* tictactoe.py: the game of Tic Tac Toe
* optimal.py: an optimal player (computing the value through min-max algorithm)
* MCTS.py: the main file, generic implementation of MCTS. Can use epsilon-greedy or UCB as simulation strategy and average, step-size, or temporal difference for backpropagation
* tests.py: a series of tests
