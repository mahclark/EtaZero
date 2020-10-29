

class Agent:
    name = "Agent"
    confidence = None

    def __init__(self, game, num=None):
        self.game = game
        if num != None:
            self.name += " " + str(num)

    def select_move(self):
        raise Exception("select_move() not implemented!")

    def set_confidence(self, x):
        self.confidence = str(int(x*1000)/10) + "%"
    
    def get_progress(self):
        return None