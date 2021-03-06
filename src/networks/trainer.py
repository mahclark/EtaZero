from time import perf_counter
import csv
import datetime
import numpy as np
import os
import torch
import utils
from agent_evaluation.arena import Arena
from agents.eta_zero import EtaZero
from agents.random_agent import RandomAgent
from agents.uct_agent import UCTAgent
from game.sevn import Game, State
from math import ceil
from networks.graph_networks import DGLValueWinNetwork
from networks.network import PolicyValueNetwork
from torch import nn


class Trainer:
    def __init__(self, model=None, load_iter=None, base_path="", section=""):
        self.base_path = base_path
        self.section = section

        if load_iter is not None:
            model = utils.load_net(load_iter, base_path, section)
            self.loaded = True
        else:
            self.loaded = False

        if not model:
            raise Exception("Model or load iteration must be provided")

        self.model = model

        self.training_data_path = os.path.join(
            base_path, "data", "training", self.section
        )

    def _default_data_generator(self, games_7=50, games_5=0, samples_per_move=50):
        data = []
        state_data = []
        labels = []

        num_games = games_7 + games_5
        remaining_7 = games_7

        eta_zero_id = EtaZero(
            self.model, training=True, samples_per_move=samples_per_move
        ).elo_id
        file_name = f"{eta_zero_id}.csv"
        data_path = os.path.join(self.training_data_path, file_name)
        print(f"Saving data at:\n{data_path}")

        print("generating data from {} games...".format(num_games))
        print(" 0%", end="")
        for i in range(num_games):
            if remaining_7 > 0:
                game_base = 7
                remaining_7 -= 1
            else:
                game_base = 5

            game = Game(game_base)

            eta_zero = EtaZero(
                self.model, training=True, samples_per_move=samples_per_move
            )
            eta_zero.set_game(game)

            while not game.over():
                game.make_move(eta_zero.select_move())

            game_str = game.state.get_game_str()

            state_strs, state_graphs, data_y = eta_zero.generate_training_labels()

            del eta_zero
            del game

            data += state_graphs
            state_data += state_strs
            labels += data_y

            with open(
                os.path.join(self.training_data_path, "game_data.csv"), "a", newline=""
            ) as game_data:
                writer = csv.writer(game_data)
                writer.writerow(
                    [
                        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                        eta_zero_id,
                        game_str,
                    ]
                )

            with open(data_path, "a", newline="") as training_data:
                writer = csv.writer(training_data)
                for x, y in zip(state_strs, data_y):

                    if isinstance(self.model, PolicyValueNetwork):
                        csv_label = y.unsqueeze(0)
                    else:
                        csv_label = y

                    writer.writerow(
                        [
                            x,  # x is game state string
                            *csv_label.tolist(),  # csv_label is pytorch tensor
                        ]
                    )

            print(" .", end="")

            # print progress
            j = 10 * (i + 1) // num_games
            if ceil(num_games * j / 10) == i + 1:
                print(f"\n{j/10:.0%}", end="")

        print()

        return self._data_loader(file_name)

    def _data_loader(self, data_file):
        with open(os.path.join(self.training_data_path, data_file)) as training_data:
            reader = csv.reader(training_data)

            data = []
            labels = []

            for row in reader:
                x = row[0]
                y = row[1:]

                if isinstance(self.model, PolicyValueNetwork):
                    data.append(State.from_str(x).to_dgl_graph(with_move_nodes=True))
                    labels.append(
                        torch.tensor(list(map(float, y[0][1:-1].split(", "))))
                    )
                else:
                    data.append(State.from_str(x).to_dgl_graph())
                    labels.append(torch.tensor(list(map(float, y))))

            return data, labels

    def count_games(self):
        """
        Counts the number of saved self-play games of the current network.
        Assumes the current network has the most recent entries in game_data.csv
        Returns 0 otherwise.
        """

        def read_elo_ids_reversed():
            with open(
                os.path.join(self.training_data_path, "game_data.csv")
            ) as game_data:

                game_data.seek(0, os.SEEK_END)
                fp = game_data.tell()

                buffer = []
                while fp >= 0:
                    game_data.seek(fp)
                    fp -= 1

                    new_byte = game_data.read(1)
                    if new_byte == "\n":
                        line = "".join(buffer[::-1])
                        if line != "":
                            yield line.split(",")[1]
                        buffer = []
                    else:
                        buffer.append(new_byte)

                line = "".join(buffer[::-1])
                if line != "":
                    yield line.split(",")[1]

        for count, elo_id in enumerate(read_elo_ids_reversed()):
            if "-".join(elo_id.split("-")[2:]) != self.model.elo_id:
                break

        return count

    @staticmethod
    def pv_loss(output, label):
        p, v = output[:-1], output[-1]
        p_lab, v_lab = label[:-1], label[-1]

        p_loss = -torch.sum(p_lab * torch.log(p + 1e-7))
        v_loss = (v - v_lab) ** 2

        return v_loss + p_loss

    def train(
        self,
        all_data,
        loss_fn=nn.MSELoss(reduction="sum"),
        n_epochs=10,
        lr=0.001,  # learning rate
        l2norm=10e-4,  # L2 norm coefficient
        batch_size=32,
        history_path="history.csv",
    ):

        if len(all_data) == 2:
            X_train, y_train = all_data
            X_val, y_val = None, None

        elif len(all_data) == 4:
            X_train, y_train, X_val, y_val = all_data

        else:
            raise Exception(
                "all_data provides unexpected number of data series ({}).".format(
                    len(all_data)
                )
            )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=l2norm
        )

        history = []

        print(f"training with {n_epochs} epochs...")
        for epoch in range(n_epochs):
            epoch_loss = 0
            self.model.train()

            batch_count = len(X_train) // batch_size
            for _ in range(batch_count):
                sample_indexes = np.random.randint(len(X_train), size=batch_size)

                for index in sample_indexes:
                    g = X_train[index]
                    label = y_train[index]

                    output = self.model.forward(g)

                    optimizer.zero_grad()
                    loss = loss_fn(output, label)
                    loss.backward()
                    output.detach()

                    optimizer.step()
                    epoch_loss += loss.detach().item()

            epoch_loss /= len(X_train)

            report = "Epoch {:05d} | ".format(epoch) + "Loss: {:.4f} | ".format(
                epoch_loss
            )

            if X_val != None:
                correct = 0
                for x, y in zip(X_val, y_val):
                    correct += self.model.forward(x)[0] * y[0] > 0
                report += "Val Acc: {:.3f}".format(correct / len(X_val))

            history.append((epoch, round(epoch_loss * 1000) / 1000))

            print(report)

        with open(
            os.path.join(self.training_data_path, history_path), "a", newline=""
        ) as history_file:
            writer = csv.writer(history_file)
            for epoch, loss in history:
                writer.writerow([self.model.elo_id, epoch, loss])

        path = self.get_save_path()
        self.save_model(path)
        print(f"Model saved:\n\tmodel: \t{self.model.elo_id}\n\tpath:  \t{path}")

    def eta_training_loop(
        self,
        loops,
        base_agent=None,
        from_train_file=None,
        samples_per_move=50,
        games_7=50,
        games_5=0,
        n_epochs=10,
        lr=0.001,
        batch_size=32,
        l2norm=10e-4,
    ):

        if not self.loaded:
            path = self.get_save_path()
            self.save_model(path)
            print(f"Model saved:\n\tmodel: \t{self.model.elo_id}\n\tpath:  \t{path}")

        for i in range(loops):
            print(
                f"\n==================== Training iteration {self.model.iteration} ({i} of {loops}) ===================="
            )

            if from_train_file == None:
                all_data = self._default_data_generator(
                    samples_per_move=samples_per_move, games_7=games_7, games_5=games_5
                )
            else:
                all_data = self._data_loader(from_train_file)
                from_train_file = None

            self.model.iterate_id()

            # if isinstance(self.model, PolicyValueNetwork):
            #     loss_fn = self.pv_loss
            # else:
            #     loss_fn = nn.MSELoss(reduction='sum')

            self.train(
                all_data, n_epochs=n_epochs, lr=lr, batch_size=batch_size, l2norm=l2norm
            )  # , loss_fn=loss_fn)

            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def save_model(self, path):
        torch.save(self.model, path)

    def get_save_path(self):
        return os.path.join(
            self.base_path, "data", "models", self.section, self.model.elo_id + ".pt"
        )

    @staticmethod
    def generate_uct_win_data(path, base=5):

        game = Game(base)

        uct1 = UCTAgent(max_evals_per_turn=500)
        uct2 = UCTAgent(max_evals_per_turn=500)

        uct1.set_game(game)
        uct2.set_game(game)

        p = uct1

        states = [game.state]

        while not game.over():
            game.make_move(p.select_move())
            states.append(game.state)

            p = uct1 if p == uct2 else uct2

        value = game.state.outcome

        writer = csv.writer(open(path, "a", newline=""))
        for state in states:
            writer.writerow([state, value])
            value *= -1


def get_win_data():
    path = os.path.join("data", "uct_win", "data.csv")
    with open(path) as data_file:
        reader = csv.reader(data_file)

        data = []
        labels = []
        for row in reader:
            state = State.from_str(row[0])
            if (
                state.outcome == 0
                and sum(
                    state.board.get_at(x, y) > -1 for x in range(5) for y in range(5)
                )
                < 10
            ):
                data.append(
                    state.to_dgl_graph(
                        with_move_nodes=isinstance(self.model, PolicyValueNetwork)
                    )
                )
                labels.append(torch.tensor([float(row[1])]))

    print(len(data))

    np.random.seed(42)
    idxs = np.random.choice(
        np.arange(0, len(data)), replace=False, size=int(len(data) * 0.8)
    )
    idxs.sort()
    val_idxs = []
    j = 0
    for i in range(len(data)):
        while j < len(idxs) - 1 and idxs[j] < i:
            j += 1

        if idxs[j] != i:
            val_idxs.append(i)
    val_idxs = np.array(val_idxs)

    def get(v, i):
        return [v[j] for j in i]

    x = get(data, idxs)
    y = get(labels, idxs)
    val = get(data, val_idxs)
    val_y = get(labels, val_idxs)

    return x, y, val, val_y


if __name__ == "__main__":

    trainer = Trainer(utils.load_net(111, section="Attempt7"), section="Attempt7")
    print(trainer.count_games())
