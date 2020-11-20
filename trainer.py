import numpy as np
import torch
from torch import nn
from dgl_value_win_network import ValueWinNetwork
from sevn import Game
from agents.random_agent import RandomAgent

def takable_label_fn(g, state):
    """
    Given a state and its graph representation, returns a tensor of predicted labels
    as (1,1) if the node is takable
    and (0,0) if the node is not takable
    """
    positions = g.ndata["position"]
    bools = [[tuple(pos) in state.board.get_takable()]*2 for pos in positions]
    return torch.tensor(bools, dtype=torch.float)

class Trainer:

    def __init__(self, model=None):

        if not model:
            model = ValueWinNetwork(
                in_dim=3,
                h_dim=10,
                out_dim=2,
                num_rels=5,
                num_hidden_layers=3
            )

        self.model = model
    
    def train(self, steps, label_fn):
        n_epochs = 10 # epochs to train
        lr = 0.01 # learning rate
        l2norm = 0 # L2 norm coefficient

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2norm)

        X_train = []
        y_train = []
        X_test = []
        y_test = []

        game = None
        agent = None

        print("generating {} data samples...".format(steps))
        for n in range(steps):
            if not game or game.over():
                game = Game(5)
                agent = RandomAgent(game)
            
            if n%5 == 0:
                X_test.append(game.state.to_dgl_graph())
                y_test.append(label_fn(X_test[-1], game.state))
            else:
                X_train.append(game.state.to_dgl_graph())
                y_train.append(label_fn(X_train[-1], game.state))
            
            game.make_move(agent.select_move())

        print("start training...")
        self.model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0
            
            for n, (g, labels) in enumerate(zip(X_train, y_train)):
                logits = self.model.forward(g)
                if logits.shape[1] != 2:
                    # TODO: fix weird bug: shape is sometimes [1,3]
                    continue
                
                optimizer.zero_grad()
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits, labels)
                loss.backward()

                optimizer.step()
                epoch_loss += loss.detach().item()
            
            epoch_loss /= n+1

            test_loss = 0

            for n, (g, labels) in enumerate(zip(X_test, y_test)):
                logits = self.model.forward(g)
                if logits.shape[1] != 2:
                    continue
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits, labels)

                test_loss += loss.detach().item()

            test_loss /= n+1

            print("Epoch {:05d} | ".format(epoch) +
                "Train Loss: {:.4f} | ".format(epoch_loss) + 
                "Test Loss: {:.4f}".format(test_loss))

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(200, takable_label_fn)

    # prove generality by training on 5x5 board and testing on 9x9 board
    game = Game(9)
    agent = RandomAgent(game)

    while not game.over():
        g = game.state.to_dgl_graph()
        h = trainer.model.forward(g)

        print()
        for row in range(game.base):
            for col in range(game.base):
                tile = game.state.board.board[row][col]
                if tile == -1:
                    print("- ",end="")
                elif (row, col) in game.state.board.get_takable():
                    print("O ",end="")
                else:
                    print("X ",end="")
            print()
        
        chosen = []
        for score, pos in zip(h, g.ndata["position"]):
            if score[0] > .5:
                chosen.append((int(pos[0]),int(pos[1])))
        
        print(chosen)

        game.make_move(agent.select_move())
