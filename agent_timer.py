import csv
import os
import time
import utils
from agents.eta_zero import EtaZero
from agents.uct_agent import UCTAgent
from collections import OrderedDict
from math import ceil
from sevn import Game


class AgentTimer:

    def __init__(self, base_path=""):
        self.timing_data_path = os.path.join(
            base_path,
            "timing_data",
            "times.csv"
        )

    def time(self, agent, base=7, num_games=1):

        print(f"timing {agent.time_id} with {num_games} games...")
        print(" 0%", end="")
        for i in range(num_games):
            game = Game(base)
            agent.set_game(game)

            while not game.over():
                remaining_tiles = game.state.num_tiles()

                start_time = time.perf_counter()
                move = agent.select_move()
                duration = time.perf_counter() - start_time

                game.make_move(move)

                with open(self.timing_data_path, "a", newline="") as timing_file:
                    writer = csv.writer(timing_file)
                    writer.writerow([
                        agent.time_id,
                        remaining_tiles,
                        f"{duration:.2f}"
                    ])

            print(" .", end="")

            # print progress
            j = 10*(i+1)//num_games
            if ceil(num_games*j/10) == i+1:
                print(f"\n{j/10:.0%}", end="")

        print()

    def avg_times(self):
        with open(self.timing_data_path) as timing_file:
            reader = csv.reader(timing_file)

            times = {}
            for row in reader:
                time_id, num_tiles, duration = row
                entry = times.setdefault(time_id, {}).setdefault(
                    int(num_tiles), [0, 0])
                entry[0] += float(duration)
                entry[1] += 1

            avg_times = {}
            for time_id, time_dict in times.items():
                avg_times[time_id] = OrderedDict(
                    sorted(zip(
                        time_dict.keys(),
                        map(lambda x: x[0]/x[1], time_dict.values())
                    ))
                )

            return avg_times


if __name__ == "__main__":
    timer = AgentTimer()
    timer.time(EtaZero(utils.load_net(4)))
    timer.time(UCTAgent(5000))
