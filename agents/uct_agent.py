import functools
from agents.agent import Agent
from math import sqrt, log
from tree_search import TreeSearch

class UCTAgent(Agent):

    name = "UCT Agent"
    C = 0.1
    n = 1000

    def __init__(self, game, num=None):
        super().__init__(game, num)
        self.visits = {}
        self.wins = {}
        self.tree_search = TreeSearch(game)

        self.move_progress = 0

    def select_move(self):
        
        best_move, score = self.tree_search.best_move_and_score(
            get_score=self._get_value
        )
        
        self.set_confidence((score*self.game.state.next_go + 1)/2)
        
        # m = 0
        # count = {}
        # for _, visits in self.visits.items():
        #     count[visits] = count.get(visits, 0) + 1
        #     if visits > m:
        #         m = visits
        
        # for i in range(m+1):
        #     if i in count:
        #         print("{0}\t{1}".format(i, count[i]))

        return best_move
    

    def _get_value(self):
        playouts = []
        for i in range(self.n):
            self.move_progress = i
            playouts.append(
            self.tree_search.playout(
                pre_fn=self._pre_fn,
                choice_fn=functools.partial(
                    self.tree_search.best_move,
                    self._get_heuristic
                ),
                record_val=self._record_state
            ))
        return sum(playouts)/self.n
    
    def _pre_fn(self):
        self.visits[self.game.state] = self.visits.get(self.game.state, 0) + 1
    
    def _record_state(self, val):
        self.wins[self.game.state] = self.wins.get(self.game.state, 0) + (val == self.game.state.next_go)
        
    def _get_heuristic(self):
        """
        Calculate the UCT heuristic value:
            win ratio + C * sqrt( ln(visits to parent) / vists to current state )
        Firstly > 0
        Then make negative if player 2
        """
        N = self.visits.get(self.game.state.parent)

        prev_player = -self.game.state.next_go

        # Return 0 if loss for previous player
        # Return 1 or -1 if win, depending on whether prev player is P1 or P2
        if self.game.state.outcome != 0:
            max_val = 1 + self.C*sqrt(log(N))
            player_won = prev_player == self.game.state.outcome
            return max_val * player_won * prev_player
        
        # Assume win ratio to be 50% for a new state
        if self.visits.get(self.game.state, 0) == 0:
            return (0.5 + self.C*sqrt(log(N))) * prev_player

        n_i = self.visits.get(self.game.state)

        # We want the win ratio for the previous player, so subtract from n_i
        s_i = n_i - self.wins.get(self.game.state, 0.5*n_i)

        return (s_i/n_i + self.C*sqrt(log(N)/n_i)) * prev_player
    
    def get_progress(self):
        return self.tree_search.get_progress(self.tree_search.progress_layers[:1] + [(self.move_progress, self.n)])