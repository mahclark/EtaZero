from agents.agent import Agent, Series
from random import choice


class RandomAgent(Agent):

    class Series(Series):

        def __init__(self):
            self.label = RandomAgent.name

        def get_members(self):
            return [RandomAgent()]

        def get_at(self, i):
            return RandomAgent()

        def __hash__(self):
            return hash(self.label)

        def __eq__(self, other):
            return other and isinstance(other, RandomAgent.Series)

    name = "Random Agent"

    def select_move(self):
        return choice(tuple(self.game.get_moves()))
