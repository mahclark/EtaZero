import csv
import json
import math
import matplotlib.pyplot as plt
import os
import portalocker
import sys
import time
import torch
import utils
import uuid
from agents.agent import NoAgent, Series
from agents.eta_zero import EtaZero
from agents.random_agent import RandomAgent
from agents.uct_agent import UCTAgent
from functools import lru_cache
from math import ceil
from sevn import Game, State
from typing import NamedTuple


class Task(NamedTuple):
    series: Series
    enemy_series: Series
    game_pairs: int
    shift: int
    elo_shift: int
    base: int = 7


class GameStats:
    """
    Representation of the play history between two agents.
    Contains number of wins and number of games played.
    Not implemented as NamedTuple so we can specify custom json encoding.
    """

    def __init__(self, wins, games):
        self.wins = wins
        self.games = games

    def __getitem__(self, i):
        return [self.wins, self.games][i]

    def __str__(self):
        return str([self.wins, self.games])

    def __repr__(self):
        return str(self)

    def update(self, wins, games):
        self.wins += wins
        self.games += games

        if self.games % 2 == 1:
            # Only odd if we faked a loss as below (since games are in pairs)
            self.games -= 1

        if self.wins == self.games:
            self.games += 1  # zero losses means inf elo, so we fake a loss


class GameStatsEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacement_map = {}

    def default(self, obj):
        if isinstance(obj, GameStats):
            self._replacement_map[id(obj)] = json.dumps(
                tuple(obj), **self.kwargs)
            return f"@@{id(obj)}@@"
        return super().default(obj)

    def encode(self, obj):
        result = super().encode(obj)
        for key, val in self._replacement_map.items():
            result = result.replace(f'"@@{key}@@"', val)
        return result


class Arena:
    """
    Battles agents and records their elo ratings and play history.
    """

    def __init__(self, base_path="", section="", saving_enabled=True):
        self.base_path = base_path
        self.elo_rating_path = os.path.join(
            base_path,
            "elo",
            section,
            "ratings.json"
        )
        self.tasks = []
        self.saving_enabled = saving_enabled

    def add_task(self, series, enemy_series, game_pairs, shift=None, base=7):
        shift_pairs = []
        try:
            for shift_val in shift:
                shift_pairs.append((shift_val, None))

        except TypeError:
            if shift is None and isinstance(enemy_series, UCTAgent.Series):
                elo_shifts = [-400, 0]
                for elo_shift in elo_shifts:
                    shift_pairs.append((None, elo_shift))
            else:
                shift_pairs.append((None, None))

        for shift, elo_shift in shift_pairs:
            self.tasks.append(Task(
                series,
                enemy_series,
                game_pairs,
                shift,
                elo_shift,
                base
            ))

    def start(self, task_wait=10):
        while True:
            no_tasks = True
            for task in self.tasks:

                shift = task.shift
                if isinstance(task.enemy_series, RandomAgent.Series):
                    shift = 0

                agents = list(enumerate(task.series.get_members()))
                if shift is not None:
                    agents = agents[shift:]

                    def get_enemy(i):
                        return task.enemy_series.get_at(i - shift), None

                elif task.elo_shift is not None:
                    # Finds the UCT with the closest elo
                    def get_enemy(i):
                        ratings, history = LockParser.read(
                            self.elo_rating_path)
                        elo_id = task.series.get_at(i).elo_id

                        enemies = []
                        played = []
                        for enemy in task.enemy_series.get_members():
                            if enemy.elo_id in ratings:
                                enemies.append((enemy, ratings[enemy.elo_id]))

                            if enemy.elo_id in history.get(elo_id, {}):
                                played.append(
                                    (enemy, history[elo_id][enemy.elo_id].games))

                        total_played = None
                        if len(played) >= 2:
                            enemy, _ = min(played, key=lambda x: x[1])
                            total_played = sum(2*(p[1]//2) for p in played)

                        else:
                            if len(enemies) == 0 or elo_id not in ratings:
                                raise NoAgent

                            enemy, _ = min(
                                enemies,
                                key=lambda e: abs(
                                    e[1] - ratings[elo_id] - task.elo_shift)
                            )

                        return enemy, total_played

                for i, agent in agents:
                    try:
                        enemy, total_played = get_enemy(i)
                    except NoAgent:
                        continue

                    _, history = LockParser.read(self.elo_rating_path)

                    if total_played is None:
                        games_played = history.get(agent.elo_id, {}).get(
                            enemy.elo_id, GameStats(0, 0)).games
                    else:
                        games_played = total_played
                    
                    if games_played < task.game_pairs*2:
                        no_tasks = False
                        self.battle(agent, enemy, min(10, task.game_pairs -
                                    games_played//2), task.base)
                        break

            if no_tasks:
                time.sleep(task_wait)

    def battle(self, agent, enemy, game_pairs=10, base=7):
        if agent.elo_id == enemy.elo_id:
            raise Exception("Agents should not have the same elo_id.")

        print(
            f"{2*game_pairs} games {agent.elo_id} vs {enemy.elo_id} (fixed):")

        wins = 0
        games = 0

        def play_game_pair(wins, games):
            game = Game(base)
            state = game.state

            outcome1 = self._play_game(state, agent, enemy)
            if outcome1 == 1:
                print(" +", end="")
            else:
                print(" -", end="")

            outcome2 = self._play_game(state, enemy, agent)
            if outcome2 == 1:
                print(" -", end="")
            else:
                print(" +", end="")

            del game
            del state

            wins += outcome1 + 1 - outcome2
            games += 2

            return wins, games

        print(" 0%", end="")

        for i in range(game_pairs):
            prev_wins = wins
            wins, games = play_game_pair(wins, games)

            rating, history = self._save(
                agent,
                enemy,
                wins=wins - prev_wins,
                games=2
            )

            j = 10*(i+1)//game_pairs
            if ceil(game_pairs*j/10) == i+1:
                print(f" won {wins} of {games}")
                print(f"{j/10:.0%}", end="")

        print()

        print(f"Won {wins} of {games}")
        if game_pairs > 0:
            print(f"New elo: {rating} (history = {history})")

    def _play_game(self, state, p1, p2):
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

    def csv_to_json(self, path):

        with open(os.path.join(self.base_path, path)) as ratings_csv:
            reader = csv.reader(ratings_csv)

            ratings = {}
            history = {}
            for row in reader:
                elo_id, rating = row[:2]
                ratings[elo_id] = float(rating)

                opponents = {}
                for s in row[2:]:
                    vs_id, wins, games = s.split("|")
                    opponents[vs_id] = GameStats(int(wins), int(games))
                history[elo_id] = opponents

        with LockParser(self.elo_rating_path) as (f, _, _):

            txt = json.dumps({
                "ratings": ratings,
                "history": history
            }, indent=3, cls=GameStatsEncoder)
            f.write(txt)

    def _save(self, agent=None, enemy=None, wins=None, games=None):
        if not self.saving_enabled:
            return

        with LockParser(self.elo_rating_path) as (f, ratings, history):

            if None not in [agent, enemy, wins, games]:

                history.setdefault(
                    agent.elo_id,
                    {}
                ).setdefault(
                    enemy.elo_id,
                    GameStats(0, 0)
                ).update(
                    wins,
                    games
                )

            ratings = self._calculate_ratings(ratings, history)

            txt = json.dumps({
                "ratings": ratings,
                "history": history
            }, indent=3, cls=GameStatsEncoder)
            f.write(txt)

            if None not in [agent, enemy, wins, games]:
                return ratings.get(agent.elo_id, None), history.get(agent.elo_id, None)

    def _calculate_ratings(self, ratings, history):
        visited = set()

        @lru_cache(maxsize=None)
        def get_rating(elo_id):
            if elo_id in visited:
                print(
                    f"Error: dependancy cycle in elo history (incl. {elo_id})")
                return None
            visited.add(elo_id)

            total_games = 0
            sum_elo = 0
            for rating, (wins, games) in [(get_rating(eid), hist) for eid, hist in history.get(elo_id, {}).items()]:
                if rating is None or 0 in [wins, games]:
                    continue

                elo = rating - 400*math.log(games/wins - 1)/math.log(10)

                total_games += games
                sum_elo += elo*games

            if total_games > 0:
                new_elo = round(sum_elo/total_games*10)/10
                ratings[elo_id] = new_elo
            else:
                new_elo = ratings.get(elo_id, None)

            return new_elo

        for eid in set([*ratings.keys(), *history.keys()]):
            get_rating(eid)

        return ratings

    def plot_all(self):
        self._save()
        plt.figure(figsize=(13, 10), facecolor="w")

        ratings, _ = LockParser.read(self.elo_rating_path)

        series_ratings = {}

        best = (0, 0)

        eta_iters = [
            int(eid.split("-")[3])
            for eid in ratings.keys()
            if eid.split("-")[0] == "EtaZero"
        ]
        max_iter = 1 if len(eta_iters) == 0 else max(eta_iters)

        uct_label_args = dict(x=max_iter, fontsize=8, va="center", ha="right")

        for elo_id, rating in ratings.items():
            split_id = elo_id.split("-")

            # UCT lines and white boxes
            if split_id[0] == "uct":
                plt.axhline(y=rating, linestyle=":")
                plt.text(y=rating, s=elo_id, c="white",
                         **uct_label_args, bbox=dict(fc="white", ec="white"))

            elif split_id[0] == "EtaZero":
                iteration = int(split_id[3])
                samples = int(split_id[1])
                xy = series_ratings.setdefault(
                    EtaZero.Series(samples), ([], []))

                xy[0].append(iteration)
                xy[1].append(rating)

                if samples == 50:
                    best = max(best, (rating, iteration))

        for elo_id, rating in ratings.items():
            split_id = elo_id.split("-")

            if split_id[0] == "uct":
                plt.text(y=rating, s=elo_id, **uct_label_args)

        for series, (x, y) in series_ratings.items():
            x, y = zip(*sorted(zip(x, y)))
            plt.plot(x, y, label=series.label)

        plt.plot([0, max(series_ratings[EtaZero.Series(50)][0])], [
                 best[0], best[0]], label=f"Iter {best[1]}: {best[0]}")

        plt.ylabel("Elo Rating")
        plt.xlabel("Training Iteration")
        plt.legend(loc="lower center")  # , bbox_to_anchor=(.95, .5))
        plt.show()


class LockParser:
    def __init__(self, file_name):
        self.file_name = file_name

        self.file = None
        self.original_contents = None

    @staticmethod
    def _create_if_not_exists(file_name):
        if not os.path.exists(file_name):
            open(file_name, 'w').close()

    @staticmethod
    def _parse(file_handle):
        file_handle.seek(0)

        if not file_handle.read(1):
            ratings = {RandomAgent().elo_id: 500}
            history = {}
        else:
            file_handle.seek(0)
            data = json.load(file_handle)
            ratings = data["ratings"]
            history = data["history"]

        for hist in history.values():
            for elo_id, game_stats in hist.items():
                hist[elo_id] = GameStats(*game_stats)

        return (
            ratings,
            history
        )

    @staticmethod
    def read(file_name):
        LockParser._create_if_not_exists(file_name)

        with open(file_name, "r") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            return LockParser._parse(f)

    def __enter__(self):
        self._create_if_not_exists(self.file_name)

        self.file = open(self.file_name, "r+")
        portalocker.lock(self.file, portalocker.LOCK_EX)
        self.original_contents = self.file.read()

        ratings, history = self._parse(self.file)
        self.file.seek(0)
        self.file.truncate()

        return (
            self.file,
            ratings,
            history
        )

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            self.file.seek(0)
            self.file.write(self.original_contents)
            self.file.truncate()

        self.file.close()
