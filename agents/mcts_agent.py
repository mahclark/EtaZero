import functools
from agents.agent import Agent
from tree_search import TreeSearch
from random import choice

class MinimaxMCTS(Agent):

    name = "Minimax MCTS"
    n = 20
    progress_layers = []

    def __init__(self, game, num=None):
        super().__init__(game, num)
        self.tree_search = TreeSearch(game)

    def select_move(self):
        """
        Performs minimax for all moves up to depth defined by minimax_depth.
        Uses Monte Carlo tree search beyond that to play games to completion.
        """
        minimax_depth = int(1 + 25/sum([self.game.get_at(row, col) > -1 for row in range(7) for col in range(7)]))
        best_move, score = self.tree_search.best_move_and_score(
            get_score=functools.partial(self.minimax, minimax_depth)
        )
        
        self.set_confidence((score*self.game.state.next_go + 1)/2)
        return best_move
    
    def minimax(self, depth):
        """
        Performs minimax on the game state tree up to the defined depth.
        Returns a score, -1 <= score <= 1, with positive value prefered by player1 and negative prefered by player 2.
        """
        if self.game.state.outcome != 0:
            return self.game.state.outcome
        
        # Perform MCTS if depth has been reached
        if depth < 1:
            return sum([
                self.tree_search.playout(self.random_choice)
                for _ in range(self.n)
            ])/self.n

        # Select the best move using this function as the choice function.
        _, best_score = self.tree_search.best_move_and_score(
            functools.partial(self.minimax, depth-1)
        )
        
        # Scale slightly so the AI prefers immediate wins and drawn-out losses.
        return best_score*0.9999
    
    def random_choice(self):
        return choice(tuple(self.game.get_moves()))
    
    def get_progress(self):
        return self.tree_search.get_progress()