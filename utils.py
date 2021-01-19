import os
import torch


def get_model_files(path="models"):
    return {
        int(path.split("-")[1]): path
        for path in os.listdir(path)
        if path[-3:] == ".pt"
    }


def load_net(i, path="models"):
    return torch.load(get_model_files(path))
