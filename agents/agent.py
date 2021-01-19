

class Agent:
    """
    An abstract class to be inherited by any agent to play the game.
    """
    name = "Agent"
    confidence = None
    playouts_played = None

    def __init__(self, num=None):
        # Used to rate agents; so elo_id should be unique for
        # every agent which may have a different performance
        self.elo_id = self.name

        # Used to measure move times; so time_id should contain
        # name and info affecting move times e.g. samples per move
        self.time_id = self.name

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
