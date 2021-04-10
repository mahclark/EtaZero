from agents.agent import Agent
import torch


class RawNetwork(Agent):
    """
    Agent which makes moves purely from guidance from a policy-value network
    """

    name = "Raw Network"

    def __init__(self, network):
        self.network = network
        self.elo_id = f"Raw-{self.network.elo_id}"

    def select_move(self):
        policy, value = self.network.evaluate(self.game.state)

        self.set_confidence((value + 1) / 2)

        return self.game.get_moves()[torch.argmax(policy).tolist()]
