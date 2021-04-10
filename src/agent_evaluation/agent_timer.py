import matplotlib.pyplot as plt
import csv
import json
import os
import socket
import sys
import time
import utils
from agents.eta_zero import EtaZero
from agents.uct_agent import UCTAgent
from collections import OrderedDict
from game.sevn import Game
from math import ceil


class TimeStats:
    """
    Representation of the move-time history of an agents.
    Contains total time and number of moves made.
    Not implemented as NamedTuple so we can specify custom json encoding.
    """

    def __init__(self, total_time, moves):
        self.total_time = total_time
        self.moves = moves

    def __getitem__(self, i):
        return [self.total_time, self.moves][i]

    def __str__(self):
        return str([self.total_time, self.moves])

    def __repr__(self):
        return str(self)

    def update(self, duration, moves):
        self.total_time = round(self.total_time + duration, 2)
        self.moves += moves


class TimeStatsEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacement_map = {}

    def default(self, obj):
        if isinstance(obj, TimeStats):
            self._replacement_map[id(obj)] = json.dumps(
                tuple(obj), **self.kwargs)
            return f"@@{id(obj)}@@"
        return super().default(obj)

    def encode(self, obj):
        result = super().encode(obj)
        for key, val in self._replacement_map.items():
            result = result.replace(f'"@@{key}@@"', val)
        return result


class AgentTimer:

    def __init__(self, base_path="", section=""):
        self.timing_data_path = os.path.join(
            base_path,
            "data",
            "timing",
            section,
            "times.json"
        )

        self.sys_id = "colab" if "google.colab" in sys.modules else socket.gethostname()

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

                self._save(agent.time_id, round(duration, 2), remaining_tiles)

            print(" .", end="")

            # print progress
            j = 10*(i+1)//num_games
            if ceil(num_games*j/10) == i+1:
                print(f"\n{j/10:.0%}", end="")

        print()

    def csv_to_json(self, path):
        n = 0
        with open(path) as timing_csv:
            reader = csv.reader(timing_csv, delimiter=",")
            for time_id, rem_tiles, duration in reader:
                n += 1
                self._save(time_id, rem_tiles, float(duration), 1)
                # if n>100:
                #     break

    def _save(self, time_id, remaining_tiles, duration, moves):
        if not os.path.exists(self.timing_data_path):
            f = open(self.timing_data_path, "w")
            f.write("{}")
            f.close()

        with open(self.timing_data_path, "r+") as timing_file:
            timing_file.seek(0)
            data = json.load(timing_file)

            for all_hist in data.values():
                for tid, hist in all_hist.items():
                    for rem_tiles, time_stats in hist.items():
                        hist[rem_tiles] = TimeStats(*time_stats)
                    all_hist[tid] = {k: v for k, v in sorted(
                        hist.items(), key=lambda x: int(x[0]))}

            data.setdefault(
                self.sys_id,
                {}
            ).setdefault(
                time_id,
                {}
            ).setdefault(
                str(remaining_tiles),
                TimeStats(0, 0)
            ).update(
                duration,
                moves
            )

            timing_file.seek(0)
            txt = json.dumps(data, indent=3, cls=TimeStatsEncoder)
            timing_file.write(txt)
            timing_file.truncate()

    def _load(self):
        with open(self.timing_data_path) as timing_file:
            data = json.load(timing_file)
            return data.get(self.sys_id, {})

    def avg_times(self):
        times = self._load()

        avg_times = {}
        for time_id, time_dict in times.items():
            avg_times[time_id] = OrderedDict(
                sorted(zip(
                    map(int, time_dict.keys()),
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
    timer.csv_to_json(
        "E:\Max Clark\Documents\cst-II\EtaZero\src\data\\timing\Attempt3\\times_pc.csv")

    # print(timer.get_info())

    for time_id, times in timer.avg_times().items():
        plt.plot(list(times.keys()), list(times.values()), label=time_id)

    plt.legend(bbox_to_anchor=(0.05, 1))
    plt.show()

    # s = State.from_str("2/cc-eac/5.bdeca.2aae.2e2.5")

    # plt.style.use("seaborn")

    # ps = []
    # vs = []
    # for i in range(max(utils.get_model_files(section="Attempt7"))):
    #     p, v = utils.load_net(i, section="Attempt7").evaluate(s)
    #     ps.append(p.tolist())
    #     vs.append(v.tolist())

    # plt.plot(vs, label="Value")

    # for p, move in zip(np.array(ps).T, s.get_moves()):
    #     plt.plot(p, label=str(move))
    # plt.legend(bbox_to_anchor=(1.05, 1))
    # plt.show()