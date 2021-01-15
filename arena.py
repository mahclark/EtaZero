import csv
import math
import os
import sys
import torch
from agents.eta_zero import EtaZero
from agents.random_agent import RandomAgent
from agents.uct_agent import UCTAgent
from math import ceil
from sevn import Game


class Arena:
    """
    This class compares the performance of two agents.
    This class is responsible for calculating and saving the elo rating of agents.
    """
    elo_rating_path = os.path.join("elo", "ratings.csv")
    default_rating = 1500
    K = 32

    def __init__(self, base_path=""):
        self.elo_rating_path = os.path.join(base_path, self.elo_rating_path)

        if not os.path.exists(self.elo_rating_path):
            open(self.elo_rating_path, 'w').close()

        with open(self.elo_rating_path) as ratings_file:
            reader = csv.reader(ratings_file)

            self.ratings = {}
            self.history = {}
            for row in reader:
                elo_id, rating = row[:2]
                self.ratings[elo_id] = float(rating)

                opponents = {}
                for s in row[2:]:
                    vs_id, wins, games = s.split("|")
                    opponents[vs_id] = (int(wins), int(games))
                self.history[elo_id] = opponents

    def battle(self, new_agent, fixed_agent, game_pairs=10, base=7):
        if new_agent.elo_id == fixed_agent.elo_id:
            raise Exception("Agents should not have the same elo_id.")

        print(
            f"{2*game_pairs} games {new_agent.elo_id} vs {fixed_agent.elo_id} (fixed):")

        wins = 0
        games = 0

        def play_game_pair(wins, games):
            game = Game(base)
            state = game.state

            outcome1 = self.play_game(state, new_agent, fixed_agent)
            outcome2 = self.play_game(state, fixed_agent, new_agent)

            del game
            del state

            wins += outcome1 + 1 - outcome2
            games += 2

            return wins, games

        for i in range(game_pairs):
            wins, games = play_game_pair(wins, games)

            j = 10*(i+1)//game_pairs
            if ceil(game_pairs*j/10) == i+1:
                print(f"{j/10:.0%} - won {wins} of {games}")

        prev_hist = self.history.get(new_agent.elo_id, {}).get(
            fixed_agent.elo_id, (0, 0))
        new_wins = prev_hist[0] + wins
        new_games = prev_hist[1] + games
        new_games += new_wins == new_games

        self.history.setdefault(new_agent.elo_id, {})[
            fixed_agent.elo_id] = (new_wins, new_games)

        total_games = 0
        sum_elo = 0
        for vs_id, (n_wins, n_games) in self.history[new_agent.elo_id].items():
            vs_elo = self.ratings[vs_id]
            elo = vs_elo - 400*math.log(n_games/n_wins - 1)/math.log(10)

            total_games += n_games
            sum_elo += elo*n_games

        new_elo = round(sum_elo/total_games*10)/10
        self.ratings[new_agent.elo_id] = new_elo

        print(f"Won {wins} of {games}")
        print(
            f"New elo: {new_elo} (history = {self.history[new_agent.elo_id]})")

        writer = csv.writer(open(self.elo_rating_path, 'w', newline=''))
        for elo_id, rating in self.ratings.items():
            hist_str = [[e] + list(map(str, h))
                        for e, h in self.history[elo_id].items()]
            history_rows = list(map("|".join, hist_str))

            writer.writerow([elo_id, f"{rating:.1f}", *history_rows])

    def get_rating(self, agent):
        return self.ratings.get(agent.elo_id, self.default_rating)

    def expected_result(self, rA, rB):
        return 1/(1 + 10**((rB - rA)/400))

    def play_game(self, state, p1, p2):
        """
        Returns the result of a game as 1 if p1 won or 0 if p2 won.
        """
        game = Game(state=state)
        p1.set_game(game)
        p2.set_game(game)
        p = p1
        while not game.over():
            game.make_move(p.select_move())
            p = p1 if p == p2 else p2

        outcome = game.state.outcome

        del game
        del state

        return (1 + outcome)//2
