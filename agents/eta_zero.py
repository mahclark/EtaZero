from agents.agent import Agent
import numpy as np
from math import sqrt

class EtaZero(Agent):

    def __init__(self, game, network, training=False, num=None):
        super().__init__(game, num)

        self.network = network
        self.training = training
        self.progress = 0
        self.tree_root = None
    
    def select_move(self):
        if self.tree_root:
            for action, child in self.tree_root.children.items():
                if child.state == self.game.state:
                    self.tree_root = self.tree_root.take_action(action)
                    break

        search_probs = self.calculate_search_probs()
        moves, prob_distr = search_probs

        if not self.training:
            move = moves[np.argmax(prob_distr)]
        else:
            move = np.random.choice(moves, p=prob_distr)

        self.tree_root = self.tree_root.take_move(move)

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

            if self.tree_root == None:
                self.tree_root = StateNode(self.network)
            
            self.probe(self.tree_root)

        sum_visits = sum([a.N**(1/tau) for a in self.tree_root.actions])
        moves = [action.move for action in self.tree_root.actions]
        probs = [a.N**(1/tau)/sum_visits for a in self.tree_root.actions]
        return (np.array(moves), np.array(probs))
    
    def probe(self, node):
        if node.is_leaf():
            if self.game.over():
                return self.game.state.outcome

            pi, v = self.network.evaluate(self.game.state)
            node.expand(self.game.state, pi)
            return v

        action = node.select_action()
        self.game.make_move(action.move)
        next_node = node.take_action(action)

        outcome = self.probe(next_node)
        self.game.undo_move()
        action.update(outcome*self.game.state.next_go)

        return outcome
    
    def get_progress(self):
        return self.progress

class StateNode:
    c_puct = 1

    def __init__(self, network, parent=None):
        self.parent = parent
        self.network = network
        self.leaf = True
    
    def expand(self, state, pi):
        self.state = state
        self.leaf = False

        assert(state.outcome == 0)

        moves, probs = pi
        assert(len(moves) == len(probs))
        self.actions = [Action(moves[i], probs[i]) for i in range(len(moves))]
        self.action_dict = { action.move : action for action in self.actions }
        self.children = { action : StateNode(self.network, self) for action in self.actions }

    def select_action(self):
        sqrt_sum_visits = sqrt(sum([a.N for a in self.actions]))
        scores = np.array([a.Q + self.c_puct*a.P*sqrt_sum_visits/(1 + a.N) for a in self.actions])
        return self.actions[np.argmax(scores)]
    
    def take_action(self, action):
        return self.children[action]
    
    def take_move(self, move):
        return self.children[self.action_dict[move]]
    
    def is_leaf(self):
        return self.leaf

class Action:
    def __init__(self, move, p):
        self.move = move
        self.P = p
        self.N = 0
        self.W = 0
        self.Q = 0
    
    def update(self, z):
        self.N += 1
        self.W += z
        self.Q = self.W/self.N
    
    def __hash__(self):
        return self.move.__hash__()


if __name__ == "__main__":
    eta = EtaZero(None, True)
    for _ in range(20):
        print(eta.select_move())