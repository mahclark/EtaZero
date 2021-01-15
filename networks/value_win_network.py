from networks.network import Network


class ValueWinNetwork(Network):

    def evaluate(self, state):
        """
        Given a game state, returns a tuple/tensor with:
            - v, the predicted win probability for the other player in range [-1, 1]
                - where 0.9 implies the current player has 5% chance of winning
                - and -0.1 implies the current player has 55% chance of winning
            - w, the predicted winner of the game in range [-1, 1]
                - where 1 implies the current player is predicted to win
                - where -1 implies the current player is predicted to lose
        """
        raise Exception("Function evaluate() not implemented!")
