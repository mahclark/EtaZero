from agents.agent import Agent
import numpy as np
import torch
from math import sqrt
from networks.value_win_network import ValueWinNetwork
from networks.policy_value_network import PolicyValueNetwork
from networks.dummy_networks import DummyPVNetwork, DummyVWNetwork
from dgl_value_win_network import DGLValueWinNetwork
from sevn import Game, State
from collections import deque, namedtuple
from tqdm import tqdm

class EtaZero(Agent):

    expected_network_types = [
        PolicyValueNetwork,
        ValueWinNetwork
    ]

    def __init__(self, game, network, training=False, num=None):
        super().__init__(game, num)

        self.network = network
        self.training = training
        self.progress = 0
        self.tree_root = None
        self.game_depth = 1

        for network_type in self.expected_network_types:
            if isinstance(network, network_type):
                self.network_type = network_type
                return
        
        raise Exception("EtaZero instantiated with unexpected network type {0}. Expected one of {1}"
                .format(network.__class__.__name__, ", ".join(map(lambda c: c.__name__, self.expected_network_types))))

    
    def select_move(self):
        if not self.training and self.tree_root: # TODO: fix node finding for when playing a different agent
            for action in self.tree_root.actions:
                if action.next_state.state == self.game.state:
                    self.tree_root = self.tree_root.take_action(action)
                    break
        
        if not self.tree_root:
            if self.network_type == PolicyValueNetwork:
                self.tree_root = StateNode()
            else:
                _, win_pred = self.network.evaluate(self.game.state)
                self.tree_root = StateNode(win_pred=win_pred)

        move_probs = self.calculate_search_probs(n=30)
        moves, prob_distr = move_probs

        if not self.training:
            move = moves[np.argmax(prob_distr)]
        else:
            move = np.random.choice(moves, p=prob_distr)
        
        if self.tree_root.parent == None: # top of the tree
            total_W = sum([action.W for action in self.tree_root.actions])
            total_N = sum([action.N for action in self.tree_root.actions])

            self.tree_root.Q = -total_W/total_N

        self.tree_root = self.tree_root.take_move(move)
        self.game_depth += 1

        return move
    
    def calculate_search_probs(self, n=1600, tau=1):
        """
        Returns a list of tuples, (move, probability) for every valid move
        The sum of all probabilities must be 1
        """

        if self.game.over():
            raise Exception("Cannot calculate search probabilities if game has finished.")

        for i in range(n):
            self.progress = i/n
            self.probe(self.tree_root)

        sum_visits = sum([a.N**(1/tau) for a in self.tree_root.actions])
        moves = [action.move for action in self.tree_root.actions]
        probs = [a.N**(1/tau)/sum_visits for a in self.tree_root.actions]
        return (np.array(moves), np.array(probs))
    
    def probe(self, node):
        if node.is_leaf():
            if self.game.over():
                return self.game.state.outcome*self.game.state.next_go

            if self.network_type == PolicyValueNetwork:
                pi, win_pred = self.network.evaluate(self.game.state)
                node.expand(self.game.state, pi)
            else:
                win_pred = node.win_pred
                node.expand_val_win(self.network, self.game)

            return win_pred

        action = node.select_action()
        self.game.make_move(action.move)

        outcome = self.probe(action.next_state)

        self.game.undo_move()
        action.update(outcome)

        return -outcome
    
    def generate_training_labels(self):
        assert self.game.over()

        z = self.game.state.outcome

        if self.game_depth%2 != 0:
            z = -z

        data_q = deque()
        if self.network_type == PolicyValueNetwork:
            raise NotImplementedError("get_training_labels() not implemented for a policy-value network")
        else:
            node = self.tree_root.parent # we don't train on a terminating node
            while node:
                data_q.appendleft((node.state, torch.tensor([node.Q, z])))
                node = node.parent
                z = -z
        
        return data_q
    
    def get_progress(self):
        return self.progress

class StateNode:
    c_puct = 1

    def __init__(self, parent=None, win_pred=None):
        self.parent = parent
        self.leaf = True
        self.win_pred = win_pred
    
    def expand(self, state, pi):
        self.state = state
        self.leaf = False

        assert(state.outcome == 0)

        moves, probs = pi
        assert(len(moves) == len(probs))
        self.actions = [Action(moves[i], probs[i], StateNode(self)) for i in range(len(moves))]
        self.action_dict = { action.move : action for action in self.actions }

    def expand_val_win(self, network, game):
        self.state = game.state
        self.leaf = False

        assert(game.state.outcome == 0)

        action_data = []
        vals = []
        for move in game.get_moves():
            game.make_move(move)
            if game.over():
                val, win = 1, 1
            else:
                val, win = network.evaluate(game.state)
            action_data.append((move, StateNode(self, win_pred=win)))
            vals.append(val)
            game.undo_move()
        
        vals = torch.tensor(vals)
        ps = vals/torch.sum(vals) # normalise vals to sum to 1
        
        self.actions = []
        self.action_dict = {}
        self.children = {}
        
        for p, (move, node) in zip(ps, action_data):
            action = Action(move, p=p, next_state=node)

            self.actions.append(action)
            self.action_dict[move] = action

    def select_action(self):
        sqrt_sum_visits = sqrt(sum([a.N for a in self.actions]))
        scores = np.array([a.Q + self.c_puct*a.P*sqrt_sum_visits/(1 + a.N) for a in self.actions])
        return self.actions[np.argmax(scores)]
    
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
    
    def update(self, z):
        self.N += 1
        self.W += z
        self.Q = self.W/self.N
        self.next_state.Q = self.Q
    
    def __hash__(self):
        return self.move.__hash__()

import sevn
import time

if __name__ == "__main__":
    net = DGLValueWinNetwork()
    dnet = DummyVWNetwork()

    t = time.perf_counter()
    move_count = 0

    for _ in range(1):
        game = Game(3)#.from_str("1/aaaaa/cabcd.aedeb.dcbee.ebdac.dacba")
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
    print("num states:", sevn.g_count)
    # for t in [net.t0, net.t1, net.t2, net.t3]:
    #     avg = sum(t)/len(t)
    #     print("{:.4f} ms".format(1000*avg))
    
    for label, move in zip(eta.generate_training_labels(), moves):
        print(label, move)