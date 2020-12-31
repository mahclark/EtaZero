import torch
import uuid

class Network:

    def __init__(self):
        super().__init__()
        self.refresh_id()
    
    def refresh_id(self):
        self.elo_id = type(self).__name__ + str(uuid.uuid4())
