import csv
import os
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
            for row in reader:
                key = row[0]
                self.ratings[key] = float(row[1])
        
    def battle(self, agentA, agentB, game_pairs=10, base=7):
        if agentA.elo_id == agentB.elo_id:
            raise Exception("Agents should not have the same elo_id.")
                
        rA = self.get_rating(agentA)
        rB = self.get_rating(agentB)
    
        for _ in range(game_pairs):
            state = Game(base).state

            # expected results after 2 games
            eA = 2*self.expected_result(rA, rB)
            eB = 2 - eA

            outcome1 = self.play_game(state, agentA, agentB)
            outcome2 = self.play_game(state, agentB, agentA)

            rA += self.K*(outcome1 + 1 - outcome2 - eA)
            rB += self.K*(outcome2 + 1 - outcome1 - eB)
        
        self.ratings[agentA.elo_id] = rA
        self.ratings[agentB.elo_id] = rB
        
        writer = csv.writer(open(self.elo_rating_path, 'w', newline=''))
        for elo_id, rating in self.ratings.items():
            writer.writerow([elo_id, rating])
    
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
        
        return (1 + game.state.outcome)//2

import time
if __name__ == "__main__":
    arena = Arena()
    a1 = RandomAgent()
    a2 = UCTAgent(max_evals_per_turn=500)
    a3 = UCTAgent(max_evals_per_turn=999)
    a4 = UCTAgent(max_evals_per_turn=9999)

    agents = [a1,a2,a3,a4]

    c = 0
    for _ in range(10):
        for i in range(len(agents)):
            for j in range(i):
                t = time.perf_counter()
                arena.battle(agents[i], agents[j], game_pairs=1, base=5)
                print(time.perf_counter() - t)
                c += 1
                print("played {} games".format(c))