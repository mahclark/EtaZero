import csv
import datetime
import numpy as np
import os
import torch
from torch import nn
from tqdm import tqdm
from networks.dgl_value_win_network import DGLValueWinNetwork
from sevn import Game, State
from agents.random_agent import RandomAgent
from agents.eta_zero import EtaZero
from agents.uct_agent import UCTAgent

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

    def _default_data_generator(self, num_games=50, game_base=7):
        data = []
        labels = []

        print("generating data from {} games...".format(num_games))
        for _ in tqdm(range(num_games)):
            game = Game(game_base)
            eta_zero = EtaZero(self.model, training=True, samples_per_move=20)
            eta_zero.set_game(game)
            
            while not game.over():
                game.make_move(eta_zero.select_move())
            
            data_x, data_y = eta_zero.generate_training_labels()

            data += data_x
            labels += data_y
        
        return data, labels
    
    def train(
        self,
        loss_fn=nn.MSELoss(reduction='sum'),
        n_epochs=10,
        lr=0.001, # learning rate
        l2norm=10e-4, # L2 norm coefficient
        batch_size=32,
        data_fn=None,
        history_path=None):

        if data_fn == None:
            data_fn = self._default_data_generator

        all_data = data_fn()

        if len(all_data) == 2:
            X_train, y_train = all_data
            X_val, y_val = None, None

        elif len(all_data) == 4:
            X_train, y_train, X_val, y_val = all_data

        else:
            raise Exception("Data function produced unexpected number of data series ({}).".format(len(all_data))) 

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2norm)

        history = []

        print("start training...")
        for epoch in range(n_epochs):
            epoch_loss = 0
            self.model.train()

            batch_count = int(len(X_train) / batch_size)
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

            report = "Epoch {:05d} | ".format(epoch) + \
                "Loss: {:.4f} | ".format(epoch_loss)
            
            if X_val != None:
                correct = 0
                for x, y in zip(X_val, y_val):
                    correct += self.model.forward(x)[0]*y[0] > 0
                report += "Val Acc: {:.3f}".format(correct/len(X_val))

                history.append((epoch, "{:.4f}".format(epoch_loss), "{:.3f}".format(correct/len(X_val))))

            print(report)
        
        if history_path != None:
            writer = csv.DictWriter(open(history_path, 'a', newline=''), fieldnames=["Epoch","MSELoss","ValAcc"])
            for epoch, loss, acc in history:
                writer.writerow({"Epoch":epoch, "MSELoss":loss, "ValAcc":acc})
        
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
        
        writer = csv.writer(open(path, 'a', newline=''))
        for state in states:
            writer.writerow([state, value])
            value *= -1

if __name__ == "__main__":
    path = "uct_win_data/data.csv"
    
    # for i in range(100):
    #     Trainer.generate_uct_win_data(path, base=5)
    
    def get_win_data():
        with open(path) as data_file:
            reader = csv.reader(data_file)

            data = []
            labels = []
            for row in reader:
                state = State.from_str(row[0])
                if state.outcome == 0 and sum(state.board.get_at(x,y) > -1 for x in range(5) for y in range(5)) < 10:
                    data.append(state.to_dgl_graph())
                    labels.append(torch.tensor([float(row[1])]))
        
        print(len(data))

        np.random.seed(42)
        idxs = np.random.choice(np.arange(0, len(data)), replace=False, size=int(len(data)*.8))
        idxs.sort()
        val_idxs = []
        j = 0
        for i in range(len(data)):
            while j < len(idxs)-1 and idxs[j] < i:
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
    
    model = DGLValueWinNetwork(dims=[3,10,10,1])
    trainer = Trainer(model=model)#, load_path="models/2020-12-18-23-06-15.pt")
    trainer.train(data_fn=get_win_data, n_epochs=20, history_path="uct_win_data/train_history.csv")
