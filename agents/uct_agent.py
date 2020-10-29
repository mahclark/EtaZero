import functools
from agents.agent import Agent
from math import sqrt, log
from tree_search import TreeSearch

class UCTAgent(Agent):

    name = "UCT Agent"
    C = 20
    n = 1000

    def __init__(self, game, num=None):
        super().__init__(game, num)
        self.visits = {}
        self.scores = {}
        self.N = 0 # total number of states simulated (not distinct)
        self.tree_search = TreeSearch(game)

    def select_move(self):
        best_move, score = self.tree_search.best_move_and_score(
            get_score=self._get_value
        )
        
        self.set_confidence((score*self.game.state.next_go + 1)/2)
        return best_move
    

    def _get_value(self):
        return sum([
            self.tree_search.playout(
                pre_fn=self._pre_fn,
                choice_fn=functools.partial(
                    self.tree_search.best_move,
                    self._get_heuristic
                ),
                record_val=self._record_state
            )
            for _ in range(self.n)
        ])/self.n
    
    def _pre_fn(self):
        self.N += 1
    
    def _record_state(self, val):
        self.visits[self.game.state] = self.visits.get(self.game.state, 0) + 1
        self.scores[self.game.state] = self.scores.get(self.game.state, 0) + val
        
    def _get_heuristic(self):
        if self.game.state.outcome != 0:
            return self.game.state.outcome*(1 + self.C*sqrt(log(self.N)))
        
        if self.visits.get(self.game.state, 0) == 0:
            return self.C*sqrt(log(self.N))

        n_i = self.visits.get(self.game.state)
        s_i = self.scores.get(self.game.state, 0)

        return s_i/n_i + self.C*sqrt(log(self.N)/n_i)
    
    def get_progress(self):
        return self.tree_search.get_progress()