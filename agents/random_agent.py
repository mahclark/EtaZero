from agents.agent import Agent
from random import choice


class RandomAgent(Agent):
    name = "Random Agent"

    def select_move(self):
        return choice(tuple(self.game.get_moves()))
