import emoji
import numpy as np


def print_map(map):
    print("Load Map:")
    for i in range(len(map)):
        for j in range(len(map[0])):
            if map[i][j] == 'X':
                out = emoji.emojize(':white_large_square:', use_aliases=True)
            elif map[i][j] == 'S':
                out = emoji.emojize(':blue_car:', use_aliases=True)
            elif map[i][j] == 'T':
                out = emoji.emojize(':fuelpump:', use_aliases=True)
            elif map[i][j] == 'G':
                out = emoji.emojize(':checkered_flag:', use_aliases=True)
            else:
                out = emoji.emojize(':black_large_square:', use_aliases=True)

            print(out + ' ', end="")
        print()


def load_map(path="./map"):
    #result = [['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
    #          ['X', 'S', '.', 'X', '.', '.', '.', '.', '.', 'X'],
    #          ['X', '.', '.', '.', '.', '.', 'X', '.', '.', 'X'],
    #          ['X', '.', '.', '.', 'T', '.', 'X', '.', 'G', 'X'],
    #          ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']]
    result = []
    with open(path) as file:
        for line in file:
            line = list(line.rstrip())
            result.append(list(line))
    return result


class Environment:
    X_COST = 3  # Costs for touching the wall. Problem with values < 3.
    T_COST = -1  # Costs / Reward for driving through the gas station.
    G_COST = -10  # Costs / Reward for arriving at the goal.
    TRANS_COST = 1  # General transition costs.

    NUM_ACTIONS = 4
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, path):
        self.map = load_map()
        print_map(self.map)

        self.maxY = len(self.map)
        self.maxX = len(self.map[0])
        self.mapShape = [self.maxY, self.maxX]
        self.num_states = np.prod(self.mapShape)
        self.P = {}
        grid = np.arange(self.num_states).reshape(self.mapShape)
        it = np.nditer(grid, flags=['multi_index'])

        # Create transitions for all states.
        while not it.finished:
            current_state = it.iterindex
            y, x = it.multi_index

            self.P[current_state] = {action: [] for action in range(self.NUM_ACTIONS)}

            if self.is_terminal(current_state):
                # No way out of a terminal state. 100% chance of staying in it, without further costs.
                self.P[current_state][self.UP] = [(0.0, current_state, 0, True)]
                self.P[current_state][self.RIGHT] = [(0.0, current_state, 0, True)]
                self.P[current_state][self.DOWN] = [(0.0, current_state, 0, True)]
                self.P[current_state][self.LEFT] = [(0.0, current_state, 0, True)]
            else:
                # Not in a terminal state.
                up_state = current_state if y == 0 else current_state - self.maxX
                right_state = current_state if x == (self.maxX - 1) else current_state + 1
                down_state = current_state if y == (self.maxY - 1) else current_state + self.maxX
                left_state = current_state if x == 0 else current_state - 1

                self.P[current_state][self.UP] = [(self.probability(current_state, self.UP, up_state),
                                                   up_state,
                                                   self.costs(current_state, self.UP, up_state),
                                                   self.is_terminal(up_state))]
                self.P[current_state][self.RIGHT] = [(self.probability(current_state, self.RIGHT, right_state),
                                                      right_state,
                                                      self.costs(current_state, self.RIGHT, right_state),
                                                      self.is_terminal(right_state))]
                self.P[current_state][self.DOWN] = [(self.probability(current_state, self.DOWN, down_state),
                                                     down_state,
                                                     self.costs(current_state, self.DOWN, down_state),
                                                     self.is_terminal(down_state))]
                self.P[current_state][self.LEFT] = [(self.probability(current_state, self.LEFT, left_state),
                                                     left_state,
                                                     self.costs(current_state, self.LEFT, left_state),
                                                     self.is_terminal(left_state))]

            it.iternext()

    # Translating the state iterator into map-coordinates.
    def stateindex_to_coordinates(self, index):
        y = int(index / self.maxX)
        x = index % self.maxX
        return y, x

    def is_terminal(self, state):
        y, x = self.stateindex_to_coordinates(state)
        return self.map[y][x] == 'G'

    def is_wall(self, state):
        y, x = self.stateindex_to_coordinates(state)
        return self.map[y][x] == 'X'

    def is_gas_station(self, state):
        y, x = self.stateindex_to_coordinates(state)
        return self.map[y][x] == 'T'

    def is_start(self, state):
        y, x = self.stateindex_to_coordinates(state)
        return self.map[y][x] == 'S'

    # Are the costs dependent on the action or the outcome?
    def costs(self, current_state, action, next_state):
        costs = self.TRANS_COST

        if self.is_terminal(current_state):
            costs = 0  # Absorbent Terminal state.
        elif self.is_wall(next_state):
            costs = self.X_COST  # Wall collision.
        elif self.is_gas_station(next_state) and not self.is_gas_station(current_state):
            costs = self.T_COST  # Arriving at gas station. (Only going through)
        elif self.is_terminal(next_state) and not self.is_terminal(current_state):
            costs = self.G_COST  # Arriving at Terminal state.

        return costs

    def probability(self, current_state, action, next_state):
        prob = 0.8

        if self.is_wall(current_state):
            prob = 0.0

        return prob
