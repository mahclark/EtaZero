import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from dgl_value_win_network import ValueWinNetwork
from sevn import Game
from agents.random_agent import RandomAgent
from agents.eta_zero import EtaZero

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
    
    def train(
        self,
        label_fn,
        loss_fn=nn.MSELoss(),
        n_epochs=5,
        lr=0.001, # learning rate
        l2norm=0, # L2 norm coefficient
        batch_size=64,
        num_games=640):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2norm)

        data = []
        labels = []

        game = None
        agent = None

        print("generating {} data samples...".format(num_games))
        for _ in tqdm(range(num_games)):
            if not game or game.over():
                game = Game(5)
                agent = RandomAgent(game, self.model)#), training=True)
            
            g = game.state.to_dgl_graph()
            data.append(g)
            labels.append(label_fn(g, game.state))
            
            game.make_move(agent.select_move())

        print("start training...")
        for epoch in range(n_epochs):
            epoch_loss = 0
            self.model.train()

            batch_count = int(len(data) / batch_size)
            for _ in tqdm(range(batch_count)):
                sample_indexes = np.random.randint(len(data), size=batch_size)
                
                for index in sample_indexes:
                    g = data[index]
                    label = labels[index]
                    
                    output = self.model.forward(g)
                    if output.shape[1] != 2:
                        # TODO: fix weird bug: shape is sometimes [1,3]
                        continue
                    
                    optimizer.zero_grad()
                    loss = loss_fn(output, label)
                    loss.backward()

                    optimizer.step()
                    epoch_loss += loss.detach().item()
                
            epoch_loss /= len(data)

            print("Epoch {:05d} | ".format(epoch) +
                "Loss: {:.4f} | ".format(epoch_loss))

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(takable_label_fn)

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
