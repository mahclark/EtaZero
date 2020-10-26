

class Agent:
    name = "Agent"

    def __init__(self, game, num=None):
        self.game = game
        if num != None:
            self.name += " " + str(num)

    def select_move(self):
        raise Exception("select_move() not implemented!")