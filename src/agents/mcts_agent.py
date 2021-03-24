import functools
from agents.agent import Agent
from tree_search import TreeSearch
from random import choice


class MinimaxMCTS(Agent):

    name = "Minimax MCTS"
    playouts_per_move = 2000
    progress_layers = []

    def set_game(self, game):
        super().set_game(game)
        self.tree_search = TreeSearch(game)

    def select_move(self):
        """
        Performs minimax for all moves up to depth defined by minimax_depth.
        Uses Monte Carlo tree search beyond that to play games to completion.
        """
        self.playouts_played = 0

        # minimax_depth = int(1 + 25/sum([self.game.get_at(row, col) > -1 for row in range(7) for col in range(7)]))
        best_move, score = self.tree_search.best_move_and_score(
            get_score=functools.partial(self.minimax, self.playouts_per_move)
        )

        self.set_confidence((score*self.game.state.next_go + 1)/2)
        return best_move

    def minimax(self, playouts):
        """
        Performs minimax on the game state tree up to the defined depth.
        Returns a score, -1 <= score <= 1, with positive value prefered by player1 and negative prefered by player 2.
        """
        if self.game.state.outcome != 0:
            return self.game.state.outcome

        # The number of playouts for each of the child states
        next_level_playouts = int(round(playouts/len(self.game.get_moves())))

        # Perform MCTS if we don't have enough playouts for minimax
        if next_level_playouts < 1:
            val = sum([
                self.tree_search.playout(self.random_choice)
                for _ in range(playouts)
            ])/playouts
            self.playouts_played += playouts
            return val

        # Select the best move using this function as the choice function.
        _, best_score = self.tree_search.best_move_and_score(
            functools.partial(self.minimax, next_level_playouts)
        )

        # Scale slightly so the AI prefers immediate wins and drawn-out losses.
        return best_score*0.9999

    def random_choice(self):
        return choice(tuple(self.game.get_moves()))

    def get_progress(self):
        return self.tree_search.get_progress()
