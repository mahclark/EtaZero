

class Agent:
    """
    An abstract class to be inherited by any agent to play the game.
    """
    name = "Agent"
    confidence = None
    playouts_played = None

    def __init__(self, num=None):
        self.elo_id = self.name
        if num != None:
            self.name += " " + str(num)

    def set_game(self, game):
        self.game = game

    def select_move(self):
        raise Exception("select_move() not implemented!")

    def set_confidence(self, x):
        self.confidence = f"{100*x:.1f}%"

    def get_progress(self):
        return None
