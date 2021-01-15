from networks.network import Network


class PolicyValueNetwork(Network):

    def evaluate(self, state):
        """
        Given a game state, returns:
            - pi, a vector of move probabilities
            - v, the predicted winner of the game
        """
        raise Exception("Function evaluate() not implemented!")
