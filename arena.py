import csv
import gc
import math
import os
import sys
from agents.random_agent import RandomAgent
from agents.uct_agent import UCTAgent
from sevn import Game

class Arena:
    """
    This class compares the performance of two agents.
    This class is responsible for calculating and saving the elo rating of agents.
    """
    elo_rating_path = os.path.join("elo", "ratings.csv")
    default_rating = 1500
    K = 32

    def __init__(self, path=None):
        if path:
            self.elo_rating_path = path
        
        if not os.path.exists(self.elo_rating_path):
            open(self.elo_rating_path, 'w').close()
        
        with open(self.elo_rating_path) as ratings_file:
            reader = csv.reader(ratings_file)

            self.ratings = {}
            self.history = {}
            for row in reader:
                elo_id, rating, vs_id, n_games = row
                vs_id = None if vs_id == "null" else vs_id

                self.ratings[elo_id] = float(rating)
                if vs_id:
                    self.history[elo_id] = (vs_id, int(n_games))
        
    def battle(self, new_agent, fixed_agent, game_pairs=10, base=7, dynamic=False):
        if new_agent.elo_id == fixed_agent.elo_id:
            raise Exception("Agents should not have the same elo_id.")

        vs_id, vs_games = self.history.get(new_agent.elo_id, (None, None))
        if vs_id and vs_id != fixed_agent.elo_id:
            raise Exception("Battling against a second agent is not supported.")

        rA = self.get_rating(new_agent)
        rB = self.get_rating(fixed_agent)

        wins = 0
        games = 0

        if vs_id:
            win_rate = self.expected_result(rA, rB)
            wins = win_rate*vs_games
            games = vs_games
        
        original_count = games
        
        def play_game_pair(wins, games):
            state = Game(base).state

            outcome1 = self.play_game(state, new_agent, fixed_agent)
            outcome2 = self.play_game(state, fixed_agent, new_agent)

            wins += outcome1 + 1 - outcome2
            games += 2

            return wins, games
    
        for _ in range(game_pairs):
            wins, games = play_game_pair(wins, games)
        
        while dynamic and (wins == games or wins == games//2) and games - original_count < 2*game_pairs:
            wins, games = play_game_pair(wins, games)
        
        if wins == games:
            games += 1

        win_rate = wins/games
        print(wins, games, win_rate)
        print(rB)
        self.ratings[new_agent.elo_id] = rB - 400*math.log(1/win_rate - 1)/math.log(10)
        self.history[new_agent.elo_id] = (fixed_agent.elo_id, games)
        
        writer = csv.writer(open(self.elo_rating_path, 'w', newline=''))
        for elo_id, rating in self.ratings.items():
            vs_id, n_games = self.history.get(elo_id, ("null","null"))
            writer.writerow([elo_id, rating, vs_id, n_games])
    
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

        gc.collect()
        
        return (1 + game.state.outcome)//2

from pympler import muppy, summary
import time
if __name__ == "__main__":

    for i in range(10):
        a1 = RandomAgent()#get_agents()[i]
        a2 = UCTAgent(max_evals_per_turn=2000)#get_agents()[i+1]
        p = a1

        t = time.perf_counter()

        game = Game(7)
        a1.set_game(game)
        a2.set_game(game)

        while not game.over():
            game.make_move(p.select_move())
            p = a1 if p == a2 else a2
        
        print("Game {}, time: {:.4f}".format(i, time.perf_counter() - t))
        gc.collect()
        
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        summary.print_(sum1)