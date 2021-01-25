import datetime
import torch.nn as nn


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.iteration = -1
        self.iterate_id()

    def iterate_id(self):
        self.iteration += 1
        self.elo_id = f'{type(self).__name__}-{self.iteration}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'


class ValueWinNetwork:

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


class PolicyValueNetwork:

    def evaluate(self, state):
        """
        Given a game state, returns:
            - pi, a vector of move probabilities in the sorted order defined by class Move
            - v, the predicted winner of the game
        """
        raise Exception("Function evaluate() not implemented!")
