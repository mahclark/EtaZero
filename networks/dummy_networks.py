import numpy as np
import random
from networks.network import PolicyValueNetwork, ValueWinNetwork



class DummyPVNetwork(PolicyValueNetwork):
    def evaluate(self, state):
        moves = state.get_moves()
        pi = np.array([1/len(moves) for _ in moves])

        return (pi, random.random())


class DummyVWNetwork(ValueWinNetwork):
    def evaluate(self, state):
        return (0, 0)
