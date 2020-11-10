from networks.policy_value_network import PolicyValueNetwork
import numpy as np

class DummyPVNetwork(PolicyValueNetwork):
    def evaluate(self, state):
        moves = np.array(list(state.board.get_moves()))
        probs = np.array([1/len(moves) for _ in moves])

        return ((moves, probs), 1)
