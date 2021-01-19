import functools
from agents.agent import Agent
from math import sqrt, log
from tree_search import TreeSearch


class UCTAgent(Agent):

    name = "UCT Agent"
    C = 0.5
    # playouts_per_move = 100

    def __init__(self, max_evals_per_turn=9999, num=None):
        super().__init__(num)
        self.max_evals_per_turn = max_evals_per_turn
        self.elo_id = f"uct-{self.max_evals_per_turn}"
        self.time_id = self.elo_id

    def set_game(self, game):
        super().set_game(game)
        self.visits = {}
        self.wins = {}
        self.tree_search = TreeSearch(game)

        self.turn_evals = 0

    def select_move(self):
        if self.game.state.parent != None:
            self.clean(self.game.state.parent, except_state=self.game.state)

        self.evals_per_move = self.max_evals_per_turn//len(
            self.game.get_moves())
        self.turn_evals = 0

        best_move, score = self.tree_search.best_move_and_score(
            get_score=self._get_value
        )

        self.set_confidence((score*self.game.state.next_go + 1)/2)

        self.game.make_move(best_move)
        next_state = self.game.state
        self.game.undo_move()

        self.clean(self.game.state, except_state=next_state)
        self.game.state.free(except_child=next_state)

        return best_move

    def clean(self, state, except_state=None):
        """
        Function to clean entries from visits and wins which are no longer needed.
        """
        if state == None or state == except_state:
            return

        self.visits.pop(state, None)  # Remove if exists
        self.wins.pop(state, None)

        for move in state.board.get_moves():
            if move.next_state != except_state:
                if move.next_state in self.visits or move.next_state in self.wins:
                    self.clean(move.next_state)

    def _get_value(self):
        playouts = []
        # for i in range(self.playouts_per_move):
        self.move_evals = 0
        while self.move_evals < self.evals_per_move:
            result = self.tree_search.playout(
                pre_fn=self._pre_fn,
                choice_fn=functools.partial(
                    self.tree_search.best_move,
                    self._get_heuristic
                ),
                terminate_fn=self._terminate_fn,
                record_val=self._record_state
            )
            if result:
                playouts.append(result)

        if len(playouts) == 0:
            return 0

        return sum(playouts)/len(playouts)

    def _pre_fn(self):
        self.move_evals += 1
        self.turn_evals += 1
        self.visits[self.game.state] = self.visits.get(self.game.state, 0) + 1

    def _terminate_fn(self):
        if self.move_evals >= self.evals_per_move:
            return True
        return False

    def _record_state(self, val):
        if val == None:  # We terminated the playout early and should undo changes to self.visits
            self.visits[self.game.state] -= 1
        else:
            self.wins[self.game.state] = self.wins.get(
                self.game.state, 0) + (val == self.game.state.next_go)

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
        # return self.tree_search.get_progress(self.tree_search.progress_layers[:1] + [(self.move_progress, self.playouts_per_move)])
        return self.turn_evals/self.max_evals_per_turn
