from agents.agent import Agent

class Human(Agent):
    name = "Player"

    def __init__(self, game, user_input, num=None):
        super().__init__(game, num)
        self.user_input = user_input

    def select_move(self):
        if self.user_input == None:
            raise Exception("Human agents must be initialised with 'user_input'.")

        while True:
            self.user_input.signal.wait()
            if self.user_input.terminate:
                return

            move = frozenset({
                tile
                for tile, selected in self.user_input.selected.items()
                if selected and tile in self.game.get_takable()
            })
            
            self.user_input.signal.clear()
            if move in self.game.get_moves():
                return move
            
            print("Invalid selection: " + str(self.user_input.selected))
            print("Move: " + str([tile for tile in move]))
            print("Available: " + str([list(move) for move in self.game.get_moves()]))