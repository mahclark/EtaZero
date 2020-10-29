import functools
from agents.agent import Agent
from tree_search import TreeSearch
from random import choice

class MCTS(Agent):

    name = "Minimax MCTS"
    n = 20
    progress_layers = []

    def __init__(self, game, num=None):
        super().__init__(game, num)
        self.tree_search = TreeSearch(game)

    def select_move(self):
        minimax_depth = int(1 + 25/sum([self.game.get_at(row, col) > -1 for row in range(7) for col in range(7)]))
        best_move, score = self.tree_search.best_move(
            get_score=functools.partial(self.minimax, minimax_depth)
        )
        
        self.set_confidence((score*self.game.state.next_go + 1)/2)
        return best_move
    
    def minimax(self, depth):
        if self.game.state.outcome != 0:
            return self.game.state.outcome
        
        if depth < 1:
            return sum([
                self.tree_search.playout(self.random_choice)
                for _ in range(self.n)
            ])/self.n

        _, best_score = self.tree_search.best_move(
            functools.partial(self.minimax, depth-1)
        )
        
        return best_score*0.9999
    
    def random_choice(self):
        return choice(tuple(self.game.get_moves()))
    
    def get_progress(self):
        return self.tree_search.get_progress()