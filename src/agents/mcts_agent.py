import functools
import random
from agents.agent import Agent
from game.sevn import State
from tree_search import TreeSearch
from random import choice, sample
from math import sqrt, log

def print(*a, **k):
    pass
print2 = print

class MCTS(Agent):

    def select_move(self):
        self.move_evals = 0
        for _ in range(self.samples_per_move):
            self.sample()
        
            # print2(f"N ({len(self.N)}):")
            # for k, v in self.N.items():
            #     print2(f"\t{str(k):>30} -> {v}")
            # print2(f"W ({len(self.W)})")
            # for k, v in self.W.items():
            #     print2(f"\t{str(k):>30} -> {v}")
            # print2(f"Q ({len(self.Q)})")
            # for k, v in self.Q.items():
            #     print2(f"\t{str(k):>30} -> {v}")
            # print2()
        
        return self.final_select()

    def sample(self, depth=0):
        self.move_evals += 1
        padding = "| "*depth

        print(padding + f"sampling node {self.game.state}")

        if not self.is_terminating_state():
            move = self.sample_select()
            # print2(padding + f"selected move {move}")
            self.game.make_move(move)
            evaluation = -self.sample(depth + 1)
            self.game.undo_move()

            print(padding + f"record {evaluation:>2}", end="\t")
            self.record(evaluation, move)
        else:
            evaluation = self.game.state.outcome * self.game.state.next_go
            print(padding + f"outcome: {self.game.state.outcome}")
            print(padding + f"next go: {self.game.state.next_go}")
            print(padding + f"eval:    {evaluation}")
        
        return evaluation
    
    # abstract

    def record(self, evaluation, move):
        ...

    def final_select(self):
        ...

    def is_terminating_state(self):
        ...

    def sample_select(self):
        ...

def argmax(moves, state):
    def inner(get_score):
        moves_by_score = {}
        best_score = -float("inf")

        print2(state, end="\t")
        for move in moves:
            score = get_score(move)
            moves_by_score.setdefault(score, set()).add(move)
            # print2(f"{move} {score:.2f}", end="\t")

            if score > best_score:
                best_score = score
        # print2()

        mv = random.choice([*moves_by_score[best_score]])
        # print2([*moves_by_score[best_score]], mv)
        return lambda: mv
    
    return inner


class UCT2(MCTS):

    C = 0.5

    def __init__(self, samples_per_move, num=None):
        super().__init__(num)
        self.samples_per_move = samples_per_move
        self.elo_id = f"uct2-{self.samples_per_move}"
        self.time_id = self.elo_id
        random.seed(42)

    def set_game(self, game):
        super().set_game(game)
        self.N = {}
        self.W = {}
        self.Q = {}
        self.ss = self.game.state

    def sample_select(self):

        @argmax(self.game.get_moves(), self.game.state)
        def get_best_move(move):
            key = (self.game.state, move)
            # print(f"{str(key):<30} {self.Q.get(key)}")
            self.game.make_move(move)
            if self.game.over():
                Q = self.game.state.outcome * self.game.state.next_go == -1
            else:
                Q = self.Q.get(key, 0.5)
            self.game.undo_move()

            U = self.C * sqrt(log(max(1, sum(
                self.N.get((self.game.state, mv), 0)
                for mv in self.game.get_moves()
            )) + 1) / self.N.get(key, 1))

            # print2("hmm", Q, U, max(1, sum(
            #     self.N.get((self.game.state, mv), 0)
            #     for mv in self.game.get_moves()
            # )) + 1, self.N.get(key))
            return Q + U

        # @argmax(self.game.get_moves())
        def _get_heuristic(move):
            """
            Calculate the UCT heuristic value:
                win ratio + C * sqrt( ln(visits to parent) / vists to current state )
            """
            state = self.game.state
            N = self.N.get(state.parent, 1)

            if self.game.over():
                if state.outcome != state.next_go:
                    return 1 + self.C * sqrt(log(N))
                else:
                    return 0 + self.C * sqrt(log(N) / N)

            if self.N.get(state, 0) == 0:
                win_ratio = 0.5
            else:
                win_ratio = 1 - self.W.get(state, 0) / self.N.get(state)

            return win_ratio + self.C * sqrt(log(N) / self.N.get(state, 1))
            
        return get_best_move()
        # return _get_heuristic()

    def record(self, evaluation, move):
        key = (self.game.state, move)
        self.N[key] = self.N.get(key, 0) + 1
        self.W[key] = self.W.get(key, 0) + max(0, evaluation)
        self.Q[key] = self.W[key] / self.N[key]

        print(f"{self.game.state},\t{move}")
    
    # def record(self, evaluation, move):
    #     key = self.game.state
    #     self.N[key] = self.N.get(key, 0) + 1
    #     self.W[key] = self.W.get(key, 0) + evaluation
    #     self.Q[key] = self.W[key] / self.N[key]

    def is_terminating_state(self):
        return self.game.over()

    def final_select(self):
        # N = { k:v for k, v in self.N.items() if k[0] == self.ss }
        # W = { k:v for k, v in self.W.items() if k[0] == self.ss }
        # Q = { k:v for k, v in self.Q.items() if k[0] == self.ss }

        # print(f"{N = }")
        # print(f"{W = }")
        # print(f"{Q = }")

        # print2(f"{self.move_evals = }")

        return self.sample_select()




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

        self.set_confidence((score * self.game.state.next_go + 1) / 2)
        return best_move

    def minimax(self, playouts):
        """
        Performs minimax on the game state tree up to the defined depth.
        Returns a score, -1 <= score <= 1, with positive value prefered by player1 and negative prefered by player 2.
        """
        if self.game.state.outcome != 0:
            return self.game.state.outcome

        # The number of playouts for each of the child states
        next_level_playouts = int(round(playouts / len(self.game.get_moves())))

        # Perform MCTS if we don't have enough playouts for minimax
        if next_level_playouts < 1:
            val = (
                sum(
                    [
                        self.tree_search.playout(self.random_choice)
                        for _ in range(playouts)
                    ]
                )
                / playouts
            )
            self.playouts_played += playouts
            return val

        # Select the best move using this function as the choice function.
        _, best_score = self.tree_search.best_move_and_score(
            functools.partial(self.minimax, next_level_playouts)
        )

        # Scale slightly so the AI prefers immediate wins and drawn-out losses.
        return best_score * 0.9999

    def random_choice(self):
        return choice(tuple(self.game.get_moves()))

    def get_progress(self):
        return self.tree_search.get_progress()
