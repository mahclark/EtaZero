import datetime
import torch.nn as nn


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.iteration = -1
        self.iterate_id()

    def iterate_id(self):
        self.iteration += 1
        self.elo_id = f'{type(self).__name__}-{self.iteration}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
