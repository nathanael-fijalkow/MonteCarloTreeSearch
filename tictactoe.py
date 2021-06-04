import numpy as np

SIZE = 3

class State(object):
    def __init__(self):
        # self.data is a SIZE * SIZE array where
        # 0 represents an empty position
        # 1 represents a cross (symbol for player 1)
        # 2 represents a circle (symbol for player 2)
        self.data = np.zeros((SIZE, SIZE))

        # player: whose turn it is to play from this state
        self.player = 1
        self.hash = 0

        # outcome can be
        # 1 if Player 1 wins
        # -1 if Player 2 wins
        # 0 if it's a tie
        # -2 if the game is not over
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
                    self.outcome = -1
                    return -1

            # Checks columns
            for j in range(0, SIZE):
                if all(x == 1 for x in self.data[:, j]):
                    self.outcome = 1
                    return 1
                if all(x == 2 for x in self.data[:, j]):
                    self.outcome = -1
                    return -1

            # Checks diagonals
            diag = [self.data[i,i] for i in range(0, SIZE)]
            if all(x == 1 for x in diag):
                self.outcome = 1
                return 1
            if all(x == 2 for x in diag):
                self.outcome = -1
                return -1

            anti_diag = [self.data[i,SIZE - 1 - i] for i in range(0, SIZE)]
            if all(x == 1 for x in anti_diag):
                self.outcome = 1
                return 1
            if all(x == 2 for x in anti_diag):
                self.outcome = -1
                return -1

            # Checks whether it's a tie
            data_all = [self.data[i,j] for i in range(0, SIZE) for j in range(0, SIZE)]
            if all(x != 0 for x in data_all):
                self.outcome = 0
                return 0

            # If we reached this point the game is still going on
            self.outcome = -2
            return -2

    # Prints the board
    def __repr__(self):
        out = "\n"
        for i in range(0, SIZE):
            out += '-'
            for _ in range(0, SIZE):
                out += '----'
            out += "\n"
            out += '| '
            for j in range(0, SIZE):
                if self.data[i, j] == 1:
                    token = 'x'
                elif self.data[i, j] == 2:
                    token = 'o'
                else:
                    token = ' '
                out += token + ' | '
            out += "\n"
        for _ in range(0, SIZE):
            out += '----'
        return out

    # Takes a state and returns the list of legal moves
    def legal_plays(self):
        legal = []
        for i in range(0, SIZE):
            for j in range(0, SIZE):
                if self.data[i, j] == 0:
                    legal.append((i,j))
        return legal

    # Hashes can also be computed recursively so we never use that function
    # def compute_hash(self):
    #     self.hash = 0
    #     for i in self.data.reshape(SIZE * SIZE):
    #         self.hash = self.hash * 3 + i
    #     return self.hash

    # Compute the hash of the state obtained by playing action (i,j)
    def compute_new_hash(self, action):
        (i, j) = action
        return self.hash + 3 ** (SIZE * i + j) * self.player

    # Returns a new state obtained by playing action (i,j)
    def next_state(self, action):
        (i, j) = action
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = self.player
        new_state.hash = self.compute_new_hash((i,j))
        new_state.player = 3 - self.player
        return new_state

    # Updates the state by playing action (i,j)
    def update_state(self, action):
        (i, j) = action
        self.data[i, j] = self.player
        self.hash = self.compute_new_hash((i,j))
        self.player = 3 - self.player
        self.outcome = 2
