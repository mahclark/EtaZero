from agents.agent import Agent


class RawNetwork(Agent):

    name = "Raw Network"

    def __init__(self, network):
        self.network = network
        self.elo_id = f"Raw-{self.network.elo_id}"

    def select_move(self):

        best = None
        best_score = -float("inf")
        for move in self.game.get_moves():
            self.game.make_move(move)
            score, _ = self.network.evaluate(self.game.state)
            self.game.undo_move()

            if score > best_score:
                best = move
                best_score = score

            print(f"{move}: {(score + 1)/2:.1%}")

        self.set_confidence((best_score + 1)/2)

        return best
