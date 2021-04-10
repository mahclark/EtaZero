import functools
import random
from agents.agent import Agent, Series
from math import sqrt, log
from tree_search import TreeSearch


class UCTAgent(Agent):
    class Series(Series):

        label = "UCT"

        all_samples = (50, 100, 200, 500, 1000, 2000, 5000, 10000)

        def __init__(self, all_samples=None):
            if all_samples is None:
                all_samples = tuple(self.all_samples)

            self.members = tuple(UCTAgent(samples) for samples in all_samples)

        def get_members(self):
            return self.members

        def __hash__(self):
            return hash(self.all_samples)

        def __eq__(self, other):
            return (
                other
                and isinstance(other, Series)
                and self.all_samples == other.all_samples
            )

    name = "UCT Agent"
    C = 0.5

    def __init__(self, samples_per_move, num=None):
        super().__init__(num)
        self.samples_per_move = samples_per_move
        self.elo_id = f"uct-{self.samples_per_move}"
        self.time_id = self.elo_id
        random.seed(42)

    def set_game(self, game):
        super().set_game(game)
        self.visits = {}
        self.wins = {}

    def select_move(self):
        if self.game.state.parent is not None:
            self.clean(self.game.state.parent, except_state=self.game.state)

        self.move_evals = 0
        while self.move_evals < self.samples_per_move:
            self.playout()

        move = self._select()

        self.game.make_move(move)
        next_state = self.game.state
        self.game.undo_move()

        if self.visits.get(next_state, 0) > 0 and next_state in self.wins:
            self.set_confidence(1 - self.wins[next_state] / self.visits[next_state])

        self.clean(self.game.state, except_state=next_state)
        self.game.state.free(except_child=next_state)

        return move

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

    def playout(self):
        self.visits[self.game.state] = self.visits.get(self.game.state, 0) + 1
        self.move_evals += 1

        if self.move_evals > self.samples_per_move:
            return None

        move = self._select()

        self.game.make_move(move)

        if self.game.over():
            v = self.game.state.outcome * self.game.state.next_go
            self.visits[self.game.state] = self.visits.get(self.game.state, 0) + 1
            self.wins[self.game.state] = self.wins.get(self.game.state, 0) + (v == 1)
        else:
            v = self.playout()

        self.game.undo_move()

        if v is None:
            self.visits[self.game.state] -= 1
            if self.visits[self.game.state] == 0:
                del self.visits[self.game.state]
        else:
            v *= -1

            self.wins[self.game.state] = self.wins.get(self.game.state, 0) + (v == 1)

        return v

    def _select(self):
        scores = []
        p = []

        for move in self.game.get_moves():
            self.game.make_move(move)
            score = self._get_heuristic()
            p.append(
                (
                    self.wins.get(self.game.state, "-"),
                    self.visits.get(self.game.state, "-"),
                    score,
                )
            )

            self.game.undo_move()

            scores.append((score, move))

        best_score, _ = max(scores)
        best_moves = [move for score, move in scores if score == best_score]

        return random.choice(best_moves)

    def _get_heuristic(self):
        """
        Calculate the UCT heuristic value:
            win ratio + C * sqrt( ln(visits to parent) / vists to current state )
        """
        state = self.game.state
        N = self.visits.get(state.parent, 1)

        if self.game.over():
            if state.outcome != state.next_go:
                return 1 + self.C * sqrt(log(N))
            else:
                return 0 + self.C * sqrt(log(N) / N)

        if self.visits.get(state, 0) == 0:
            win_ratio = 0.5
        else:
            win_ratio = 1 - self.wins.get(state, 0) / self.visits.get(state)

        return win_ratio + self.C * sqrt(log(N) / self.visits.get(state, 1))

    def get_progress(self):
        return self.move_evals / self.samples_per_move
