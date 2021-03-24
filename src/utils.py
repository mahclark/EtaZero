import os
import torch


def get_model_files(base_path="", section=""):
    return {
        int(path.split("-")[1]): path
        for path in os.listdir(os.path.join(
            base_path,
            "models",
            section)
        )
        if path[-3:] == ".pt"
    }


def load_net(i, base_path="", section=""):
    return torch.load(os.path.join(
        base_path,
        "models",
        section,
        get_model_files(base_path, section)[i]
    ))
