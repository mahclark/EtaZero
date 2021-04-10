import numpy as np
import pygame
import time
import torch
from agents.agent import Agent, Series
from collections import deque, namedtuple
from game.renderer import Renderer
from game.sevn import Game, State
from math import sqrt
from networks.graph_networks import DGLValueWinNetwork
from networks.dummy_networks import DummyPVNetwork, DummyVWNetwork
from networks.network import PolicyValueNetwork, ValueWinNetwork
from threading import Thread
from tqdm import tqdm
from utils import get_model_files, load_net


class EtaZero(Agent):
    class Series(Series):
        def __init__(
            self, samples_per_move, base_path="", section="", lower_limit=None
        ):
            self.label = f"{EtaZero.name}-{samples_per_move}"
            self.samples_per_move = samples_per_move
            self.base_path = base_path
            self.section = section
            self.lower_limit = lower_limit

        def get_members(self):
            return [
                EtaZero(
                    load_net(i, self.base_path, self.section), self.samples_per_move
                )
                for i, _ in sorted(
                    get_model_files(self.base_path, self.section).items()
                )
                if self.lower_limit is None or i >= self.lower_limit
            ]

        def __hash__(self):
            return hash(self.samples_per_move)

        def __eq__(self, other):
            return (
                other
                and isinstance(other, Series)
                and self.samples_per_move == other.samples_per_move
            )

    name = "EtaZero"

    expected_network_types = [PolicyValueNetwork, ValueWinNetwork]

    def __init__(self, network, training=False, samples_per_move=50, num=None):
        super().__init__(num)

        self.network = network
        self.training = training
        self.progress = 0
        self.samples_per_move = samples_per_move
        self.elo_id = self.make_elo_id(samples_per_move, self.network.elo_id)
        self.time_id = f"{self.name}-{samples_per_move}"

        self.network_type = None
        for network_type in self.expected_network_types:
            if isinstance(network, network_type):
                self.network_type = network_type

        if not self.network_type:
            raise Exception(
                "EtaZero instantiated with unexpected network type {0}. Expected one of {1}".format(
                    network.__class__.__name__,
                    ", ".join(map(lambda c: c.__name__, self.expected_network_types)),
                )
            )

    @staticmethod
    def make_elo_id(samples_per_move, network_id):
        return f"{EtaZero.name}-{samples_per_move}-{network_id}"

    def set_game(self, game):
        super().set_game(game)

        if self.network_type == PolicyValueNetwork:
            pi, val = self.network.evaluate(self.game.state)
            self.tree_root = StateNode(self.game.state)
            self.tree_root.expand(self.game, pi, val)
        else:
            _, win_pred = self.network.evaluate(self.game.state)
            self.tree_root = StateNode(self.game.state, win_pred=win_pred)
            self.tree_root.expand_val_win(self.network, self.game)
        self.move_root = self.tree_root

    def select_move(self):
        if self.move_root.state != self.game.state:
            self.find_state()

        move_probs = self.sample_and_get_probs(n=self.samples_per_move)
        moves, prob_distr = move_probs

        if not self.training:
            move = moves[np.argmax(prob_distr)]
        else:
            np_moves = np.empty(len(moves), object)
            np_moves[:] = moves
            move = np.random.choice(np_moves, p=prob_distr)

        if self.move_root.parent == None:  # top of the tree
            total_W = sum([action.W for action in self.move_root.actions])
            total_N = sum([action.N for action in self.move_root.actions])

            self.move_root.Q = -total_W / total_N

        score = self.move_root.action_dict[move].Q
        self.set_confidence((score + 1) / 2)

        self.move_root = self.move_root.take_move(move)

        return move

    def select_move_from_state(self, state_node):
        """
        Returns the most visited move from the given state.
        Returns None if there are no visits.
        """

        if state_node.actions:
            move_probs = self.get_move_probs(state_node)
            moves, prob_distr = move_probs
            move = moves[np.argmax(prob_distr)]

            return move

        return None

    def find_state(self):
        for action in self.move_root.actions:
            if action.next_state.state == self.game.state:
                self.move_root = self.move_root.take_move(action.move)
                return

        raise Exception(
            "EtaZero couldn't find current game state. This happens when > 1 move has been made since EtaZero's last move."
        )

    def sample_and_get_probs(self, n=1600, tau=1):
        """
        Returns a list of tuples, (move, probability) for every valid move
        The sum of all probabilities must be 1
        """

        if self.game.over():
            raise Exception(
                "Cannot calculate search probabilities if game has finished."
            )

        for i in range(n):
            self.progress = i / n
            self.probe(self.move_root)

        return self.get_move_probs(self.move_root)

    def get_move_probs(self, state_node, tau=1):

        sum_visits = max(1, sum([a.N ** (1 / tau) for a in state_node.actions]))
        moves = [action.move for action in state_node.actions]
        probs = [a.N ** (1 / tau) / sum_visits for a in state_node.actions]
        return (moves, np.array(probs))

    def probe(self, node):
        if node.is_leaf():
            if self.game.over():
                return self.game.state.outcome * self.game.state.next_go

            if self.network_type == PolicyValueNetwork:
                pi, win_pred = self.network.evaluate(self.game.state)
                node.expand(self.game, pi, win_pred)
            else:
                win_pred = node.win_pred
                node.expand_val_win(self.network, self.game)

            return win_pred

        action = node.select_action()
        self.game.make_move(action.move)

        outcome = self.probe(action.next_state)

        self.game.undo_move()
        action.update(-outcome)

        return -outcome

    def generate_training_labels(self):
        assert self.game.over()

        state_strs = []
        state_graphs = []
        data_y = []

        node = self.move_root
        if node.state.outcome != 0:
            node = self.move_root.parent  # we don't train on a terminating node

        while node != None and (
            self.network_type == PolicyValueNetwork or node.Q != None
        ):

            if self.network_type == PolicyValueNetwork:
                graph = node.state.to_dgl_graph(with_move_nodes=True)

                ns = torch.tensor([float(a.N) for a in node.actions])
                policy = ns / torch.sum(ns)
                value = 1 if self.game.state.outcome == node.state.next_go else -1
                label = torch.zeros(graph.num_nodes())

                # print(len(node.state.get_moves()), policy)

                label[: len(node.state.get_moves())] = policy
                label = torch.cat((label, torch.tensor(value).unsqueeze(0)))

            else:
                graph = node.state.to_dgl_graph()

                win_label = 1 if self.game.state.outcome == node.state.next_go else -1
                label = torch.tensor([node.Q, win_label])

            state_strs.append(str(node.state))
            state_graphs.append(graph)
            data_y.append(label)

            node = node.parent

        return (state_strs, state_graphs, data_y)

    def get_progress(self):
        return self.progress


class StateNode:
    c_puct = 3

    def __init__(self, state, parent=None, win_pred=None):
        self.parent = parent
        self.leaf = True
        self.win_pred = win_pred
        self.state = state
        self.actions = None
        self.Q = None

    def expand(self, game, pi, win_pred):
        self.leaf = False
        self.win_pred = win_pred

        assert game.state.outcome == 0

        def get_child_state(move):
            game.make_move(move)
            state = game.state
            game.undo_move()
            return state

        moves = game.get_moves()
        assert len(moves) == len(pi)

        self.actions = [
            Action(move, p, StateNode(get_child_state(move), self))
            for move, p in zip(moves, pi)
        ]

        self.action_dict = {action.move: action for action in self.actions}

    def expand_val_win(self, network, game):
        self.leaf = False

        assert game.state.outcome == 0

        action_data = []
        vals = []
        for move in game.get_moves():
            game.make_move(move)
            if game.over():
                result = game.state.outcome * game.state.next_go
                val, win = -result, result
            else:
                val, win = network.evaluate(game.state)

            action_data.append((move, StateNode(game.state, self, win_pred=win)))
            vals.append(val)
            game.undo_move()

        vals = (1 + torch.tensor(vals)) / 2  # adjust range from [-1,1] to [0,1]
        ps = vals / torch.sum(vals)  # normalise vals to sum to 1

        self.actions = []
        self.action_dict = {}
        self.children = {}

        for p, (move, node) in zip(ps, action_data):
            action = Action(move, p=p, next_state=node)

            self.actions.append(action)
            self.action_dict[move] = action

    def select_action(self):
        return self.actions[np.argmax(self.get_action_scores())]

    def get_action_scores(self):
        sqrt_sum_visits = sqrt(sum([a.N for a in self.actions]))
        scores = np.array(
            [
                a.Q + self.c_puct * a.P * sqrt_sum_visits / (1 + a.N)
                for a in self.actions
            ]
        )
        return scores

    def take_move(self, move):
        return self.action_dict[move].next_state

    def is_leaf(self):
        return self.leaf


class Action:
    def __init__(self, move, p, next_state):
        self.move = move
        self.P = p
        self.N = 0
        self.W = 0
        self.Q = 0
        self.next_state = next_state
        self.next_state.Q = self.Q

    def update(self, z):
        self.N += 1
        self.W += z
        self.Q = self.W / self.N
        self.next_state.Q = self.Q

    def __hash__(self):
        return self.move.__hash__()


class EtaZeroVisualiser:
    def __init__(self, eta_zero, state_list):
        self.eta_zero = eta_zero
        self.current_node = self.eta_zero.tree_root
        self.set_actions()
        self.undo_history = deque()
        self.redo_history = deque()
        self.state_list = state_list

        pygame.init()
        x_size, y_size = 1000, 600
        self.screen = pygame.display.set_mode((x_size, y_size), pygame.RESIZABLE)
        pygame.display.set_caption("EtaZero Visualiser")

        self.h_gap = 150
        self.font = pygame.font.SysFont("Bahnschrift", 20)
        self.clock = pygame.time.Clock()

        if self.eta_zero.game.over():
            xs, _, ys = eta_zero.generate_training_labels()
        else:
            xs, ys = [], []

        if eta_zero.network_type == ValueWinNetwork:
            self.labels = {state: (val, win) for state, (val, win) in zip(xs, ys)}
        else:
            self.labels = {state: pv for state, pv in zip(xs, ys)}

        # thread = Thread(target = self.begin)
        # thread.start()
        self.begin()

    def begin(self):

        done = False
        while not done:
            mx, my = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

                if event.type == pygame.VIDEORESIZE:
                    x_size = max(600, event.w)
                    y_size = max(600, event.h)
                    self.screen = pygame.display.set_mode(
                        (x_size, y_size), pygame.RESIZABLE
                    )

                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        i = (mx - 50) // self.h_gap
                        if 0 <= i < len(self.actions):
                            self.go_down(self.actions[i][0].move)
                    elif event.button in [3, 4]:
                        self.go_up()
                    elif event.button == 5:
                        if (
                            self.current_node.state in self.state_list
                            and self.current_node.actions
                        ):
                            idx = self.state_list.index(self.current_node.state) + 1
                            if idx < len(self.state_list):
                                for action in self.current_node.actions:
                                    if action.next_state.state == self.state_list[idx]:
                                        self.go_down(action.move)
                                        break
                    elif event.button == 6:
                        if len(self.undo_history) > 0:
                            self.redo_history.append(self.current_node)
                            self.current_node = self.undo_history.pop()
                            self.set_actions()
                    elif event.button == 7:
                        if len(self.redo_history) > 0:
                            self.undo_history.append(self.current_node)
                            self.current_node = self.redo_history.pop()
                            self.set_actions()

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP:
                        self.go_up()
                    if event.key == pygame.K_DOWN:
                        move = self.eta_zero.select_move_from_state(self.current_node)
                        if move:
                            self.go_down(move)

            self.render()
            self.clock.tick(60)

    def go_up(self):
        if self.current_node.parent:
            self.undo_history.append(self.current_node)
            self.redo_history.clear()
            self.current_node = self.current_node.parent
            self.set_actions()

    def go_down(self, move):
        self.undo_history.append(self.current_node)
        self.redo_history.clear()
        self.current_node = self.current_node.take_move(move)
        self.set_actions()

    def set_actions(self):
        if self.current_node.actions:
            self.actions = sorted(
                zip(self.current_node.actions, self.current_node.get_action_scores()),
                key=lambda a: -a[0].N,
            )
        else:
            self.actions = []

    def format_move(self, move):
        return "{0},{1}".format(move.row, move.col)

    def render(self):
        self.screen.fill([20, 20, 20])

        white = (255, 255, 255)
        blue = (90, 100, 150)
        d_blue = (45, 50, 75)

        margin = 100

        state_lbl = self.font.render(str(self.current_node.state), 1, white)
        self.screen.blit(state_lbl, (margin - 27, 10))

        pygame.draw.circle(self.screen, blue, (margin, 80), 30)

        if self.current_node.win_pred is not None:
            prob_lbl_text = f"{(1 + self.current_node.win_pred)/2:.1%}"
        else:
            prob_lbl_text = "None"

        prob_lbl = self.font.render(prob_lbl_text, 1, white)
        self.screen.blit(prob_lbl, (margin - 27, 66))

        board_surf = pygame.Surface((120, 120), pygame.SRCALPHA, 32)
        Renderer.draw_board(board_surf, self.current_node.state.board)
        self.screen.blit(board_surf, (margin + 40, 37))

        base = self.current_node.state.score.base
        grid_height = 106
        grid_width = (grid_height - 1) * (2 * base + 1) // base + 1
        grid_surf = pygame.Surface((grid_width, grid_height), pygame.SRCALPHA, 32)
        Renderer.draw_score_grid(grid_surf, self.current_node.state.score)
        self.screen.blit(grid_surf, (margin + 190, 44))

        if str(self.current_node.state) in self.labels:
            if self.eta_zero.network_type == ValueWinNetwork:
                label = ", ".join(
                    map("{:.2f}".format, self.labels[str(self.current_node.state)])
                )
                title_lbl = self.font.render(f"val, win", 1, white)
                label_lbl = self.font.render(label, 1, white)
            else:
                label = f"{self.labels[str(self.current_node.state)][-1]:.2f}"
                title_lbl = self.font.render(f"val", 1, white)
                label_lbl = self.font.render(label, 1, white)

            self.screen.blit(title_lbl, (margin + 190 + grid_width + 40, 20))
            self.screen.blit(label_lbl, (margin + 190 + grid_width + 40, 66))

        if str(self.current_node.state) in self.labels:
            ps, _ = torch.sort(
                self.labels[str(self.current_node.state)], descending=True
            )
            ps = self.labels[str(self.current_node.state)]
        else:
            ps = None

        for i, (action, score) in enumerate(self.actions):
            lbl0 = self.font.render(
                " | ".join(map(self.format_move, action.move)), 1, white
            )
            self.screen.blit(lbl0, (margin + self.h_gap * i - 20, 160))

            dot_text_col = (blue, white)
            if action.next_state.state in self.state_list:
                dot_text_col = (white, d_blue)

            pygame.draw.circle(
                self.screen, dot_text_col[0], (margin + self.h_gap * i, 220), 20
            )
            if action.next_state:
                if action.next_state.win_pred is not None:
                    lbl1_text = f"{(1 + action.next_state.win_pred)/2:.1%}"
                else:
                    lbl1_text = "None"
                lbl1 = self.font.render(lbl1_text, 1, dot_text_col[1])
                self.screen.blit(lbl1, (margin + self.h_gap * i - 20, 210))

            lbl2 = self.font.render("P = {:.1%}".format(action.P), 1, white)
            self.screen.blit(lbl2, (margin + self.h_gap * i - 10, 250))

            lbl3 = self.font.render("N = {:.2f}".format(action.N), 1, white)
            self.screen.blit(lbl3, (margin + self.h_gap * i - 10, 300))

            lbl4 = self.font.render("W = {:.2f}".format(action.W), 1, white)
            self.screen.blit(lbl4, (margin + self.h_gap * i - 10, 350))

            lbl5 = self.font.render("Q = {:.1%}".format((1 + action.Q) / 2), 1, white)
            self.screen.blit(lbl5, (margin + self.h_gap * i - 10, 400))

            lbl6 = self.font.render("Sc = {:.3f}".format(score), 1, white)
            self.screen.blit(lbl6, (margin + self.h_gap * i - 10, 450))

            if self.eta_zero.network_type == PolicyValueNetwork and ps is not None:
                for j, move in enumerate(self.current_node.state.get_moves()):
                    if move == action.move:
                        break

                lbl7 = self.font.render(f"Pi = {ps[j]:.3f}", 1, white)
                self.screen.blit(lbl7, (margin + self.h_gap * i - 10, 500))

        for i, state in enumerate(self.state_list):
            col = blue if state == self.current_node.state else d_blue
            pygame.draw.circle(
                self.screen,
                col,
                (20, self.screen.get_size()[1] * (i + 1) // (len(self.state_list) + 1)),
                10,
            )

        pygame.display.flip()


if __name__ == "__main__":
    net = DGLValueWinNetwork()
    dnet = DummyVWNetwork()

    t = time.perf_counter()
    move_count = 0

    for _ in range(1):
        game = Game(3)  # .from_str("1/aaaaa/cabcd.aedeb.dcbee.ebdac.dacba")
        eta = EtaZero(game, net, training=True)
        moves = []
        while not game.over():
            move = eta.select_move()
            game.make_move(move)
            moves.append(set(move))
            move_count += 1
            # print(move_count)
            # break

    print("total: {:.4f} s".format((time.perf_counter() - t)))
    print("num moves:", move_count)

    for label, move in zip(eta.generate_training_labels(), moves):
        print(label, move)
