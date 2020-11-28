import datetime
import numpy as np
import os
import torch
from torch import nn
from tqdm import tqdm
from networks.dgl_value_win_network import DGLValueWinNetwork
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

    def __init__(self, model=None, load_path=None):
        if load_path:
            model = torch.load(load_path)

        if not model:
            model = DGLValueWinNetwork(
                dims=[3,10,10,2],
                on_cuda=True
            )

        self.model = model
        self.prev_model_path = None

    
    def train(
        self,
        loss_fn=nn.MSELoss(reduction='sum'),
        n_epochs=10,
        lr=0.001, # learning rate
        l2norm=10e-4, # L2 norm coefficient
        batch_size=32,
        num_games=640,
        game_base=5):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2norm)

        data = []
        labels = []

        print("generating data from {} games...".format(num_games))
        for _ in tqdm(range(num_games)):
            game = Game(game_base)
            eta_zero = EtaZero(game, self.model, training=True, samples_per_move=20)
            
            while not game.over():
                game.make_move(eta_zero.select_move())
            
            data_x, data_y = eta_zero.generate_training_labels()

            data += data_x
            labels += data_y

        print("start training...")
        for epoch in range(n_epochs):
            epoch_loss = 0
            self.model.train()

            batch_count = int(len(data) / batch_size)
            for _ in range(batch_count):
                sample_indexes = np.random.randint(len(data), size=batch_size)
                
                for index in sample_indexes:
                    g = data[index]
                    label = labels[index]
                    
                    output = self.model.forward(g)
                    
                    optimizer.zero_grad()
                    loss = loss_fn(output, label)
                    loss.backward()
                    output.detach()

                    optimizer.step()
                    epoch_loss += loss.detach().item()
                
            epoch_loss /= len(data)

            print("Epoch {:05d} | ".format(epoch) +
                "Loss: {:.4f} | ".format(epoch_loss))
        
        # if self.prev_model_path != None:
        #     prev_model = torch.load(self.prev_model_path)
        #     self.compare(prev_model)
        
        path = self.get_save_path()
        self.save_model(path)
        self.prev_model_path = path
        print("model saved at path:", path)
    
    def save_model(self, path):
        torch.save(self.model, path)

    @staticmethod
    def get_save_path():
        return os.path.join(
            "models",
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".pt"
        )
    
    def compare(self, model, num_games=4, game_base=5):

        for i in range(num_games):
            game = Game(game_base)

            if i % 2 == 0:
                p1 = EtaZero(game, self.model, num=1)
                p2 = EtaZero(game, model, num=2)
            else:
                p1 = EtaZero(game, model, num=1)
                p2 = EtaZero(game, self.model, num=2)

            next_player = p1
            while not game.over():
                game.make_move(next_player.select_move())
                next_player = p1 if next_player == p2 else p2
            
            if (i % 2 == 0) == (game.state.outcome == 1):
                print("Training model won as player {}".format(i % 2 + 1))
            else:
                print("Training model lost as player {}".format(i % 2 + 1))

if __name__ == "__main__":

    trainer = Trainer(load_path="models\\2020-11-28-12-01-33.pt")
    while True:
        # num_games = 20 if i == 0 else 100
        trainer.train(num_games=100, game_base=5, batch_size=8, n_epochs=20)

    # prove generality by training on 5x5 board and testing on 9x9 board
    # game = Game(9)
    # agent = RandomAgent(game)

    # while not game.over():
    #     g = game.state.to_dgl_graph()
    #     h = trainer.model.forward(g)

    #     print()
    #     for row in range(game.base):
    #         for col in range(game.base):
    #             tile = game.state.board.board[row][col]
    #             if tile == -1:
    #                 print("- ",end="")
    #             elif (row, col) in game.state.board.get_takable():
    #                 print("O ",end="")
    #             else:
    #                 print("X ",end="")
    #         print()
        
    #     chosen = []
    #     for score, pos in zip(h, g.ndata["position"]):
    #         if score[0] > .5:
    #             chosen.append((int(pos[0]),int(pos[1])))
        
    #     print(chosen)

    #     game.make_move(agent.select_move())
