from networks.network import Network

class ValueWinNetwork(Network):
    
    def evaluate(self, state):
        """
        Given a game state, returns:
            - v, the predicted win probabilityfor the next player
            - w, the predicted winner of the game
        """
        raise Exception("Function evaluate() not implemented!")
