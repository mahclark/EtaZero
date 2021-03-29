import matplotlib.pyplot as plt
import csv
import os
import time
import utils
from agents.eta_zero import EtaZero
from agents.uct_agent import UCTAgent
from collections import OrderedDict
from game.sevn import Game
from math import ceil


class AgentTimer:

    def __init__(self, base_path="", section=""):
        self.timing_data_path = os.path.join(
            base_path,
            "timing_data",
            section,
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

    def get_info(self):
        with open(self.timing_data_path) as timing_file:
            reader = csv.reader(timing_file)

            cur_tiles = {}
            num_games = {}

            for row in reader:
                time_id, num_tiles, _ = row
                num_tiles = int(num_tiles)

                if cur_tiles.setdefault(time_id, 0) <= num_tiles:
                    num_games.setdefault(time_id, 0)
                    num_games[time_id] += 1

                cur_tiles[time_id] = num_tiles

            return num_games


if __name__ == "__main__":
    import numpy as np
    from game.sevn import State

    timer = AgentTimer()
    # timer.time(
    #     EtaZero(utils.load_net(4), samples_per_move=100),
    #     num_games=1
    # )
    # timer.time(
    #     UCTAgent(10000),
    #     num_games=1
    # )

    # print(timer.get_info())

    # for time_id, times in timer.avg_times().items():
    #     plt.plot(list(times.keys()), list(times.values()), label=time_id)

    # plt.legend(bbox_to_anchor=(0.05, 1))
    # plt.show()

    s = State.from_str("2/cc-eac/5.bdeca.2aae.2e2.5")

    plt.style.use("seaborn")

    ps = []
    vs = []
    for i in range(max(utils.get_model_files(section="Attempt7"))):
        p, v = utils.load_net(i, section="Attempt7").evaluate(s)
        ps.append(p.tolist())
        vs.append(v.tolist())

    plt.plot(vs, label="Value")

    for p, move in zip(np.array(ps).T, s.get_moves()):
        plt.plot(p, label=str(move))
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()
