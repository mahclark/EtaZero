import csv
import os
import torch
import utils
from agents.random_agent import RandomAgent
from dataclasses import dataclass
from game.sevn import Game, State


class Tester:
    class LimitReached(Exception):
        pass

    @dataclass
    class SearchLimit:
        limit: int = 1000_000

        def decrement(self):
            self.limit -= 1
            if self.limit < 0:
                raise Tester.LimitReached

    def __init__(self, base_path=""):
        self.base_path = base_path
        self.data_path = os.path.join(
            base_path, "agent_evaluation", "test_examples.csv"
        )

    def generate(self, agent, n_games=100, base=5):
        with open(self.data_path, "w+") as f:
            reader = csv.reader(f)

            exm_by_tiles = {}
            for row in reader:
                state, num_tiles, lbl = row
                exm_by_tiles.setdefault(num_tiles, []).append((state, lbl))

            for n in range(n_games):

                game = Game(base)
                agent.set_game(game)

                while not game.over():
                    try:
                        lbl = self.solve(game.state)
                        lbl.append(1 if 1 in lbl else -1)

                        exm_by_tiles.setdefault(game.state.num_tiles, []).append(
                            (str(game.state), lbl)
                        )
                        print((str(game.state), lbl))
                    except Tester.LimitReached:
                        print(f"too big: {game.state.num_tiles()}, {game.state}")

                    game.make_move(agent.select_move())

    def solve(self, state, verbose=False):
        game = Game(state=state)

        sl = Tester.SearchLimit()

        policy = []

        for move in game.get_moves():
            game.make_move(move)
            policy.append(self.minimax(game, sl))
            game.undo_move()

            if verbose:
                print(f"{str(move):>20}  :  {int(policy[-1]>0)}")
        print(state.num_tiles(), self.SearchLimit.limit - sl.limit)
        return policy

    def minimax(self, game, search_limit=SearchLimit()):
        search_limit.decrement()

        if game.over():
            return -game.state.outcome * game.state.next_go

        for move in game.state.get_moves():
            game.make_move(move)
            val = self.minimax(game, search_limit)
            game.undo_move()

            if val == 1:
                return -1

        return 1

    def equal_states(self, s1, s2):
        if s1.board.base != s2.board.base:
            return False
        if s1.num_tiles != s2.num_tiles:
            return False

        nets = [utils.load_net(i) for i in range(5)]

        def round_t(a):
            return torch.round(a * 10 ** 5) / (10 ** 5)

        def compare_tensors(a, b):
            return torch.all(round_t(a).sort()[0].eq(round_t(b).sort()[0]))

        for i, net in enumerate(nets):
            eval1 = net.evaluate(s1)
            eval2 = net.evaluate(s2)

            if not compare_tensors(eval1[0], eval2[0]):
                print(i)
                # print(eval1[0].tolist())
                # print(eval2[0].tolist())
                print(eval1[0].sort()[0].tolist())
                print(eval2[0].sort()[0].tolist())
                print(eval1[0].sort()[0].eq(eval2[0].sort()[0]))
                return False
            if not compare_tensors(eval1[0], eval2[0]):
                print(eval1)
                print(eval2)
                return False

        return True


if __name__ == "__main__":
    tester = Tester()

    # tester.generate(RandomAgent(), 1)
    tester.solve(State.from_str("1/-b-cbgda-g/7.7.7.1eegf2.4d2.4b2.7"), verbose=True)
