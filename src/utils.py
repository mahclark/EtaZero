import os
import torch


def get_model_files(base_path="", section=""):
    return {
        int(path.split("-")[1]): path
        for path in os.listdir(os.path.join(base_path, "data", "models", section))
        if path[-3:] == ".pt"
    }


def get_training_files(base_path="", section=""):
    return {
        int(path.split("-")[3]): path
        for path in os.listdir(os.path.join(base_path, "data", "training", section))
        if path[-4:] == ".csv" and path[:8] == "EtaZero-"
    }


def load_net(i, base_path="", section=""):
    return torch.load(
        os.path.join(
            base_path, "data", "models", section, get_model_files(base_path, section)[i]
        )
    )


def print_latex_board(state):
    cols = ["yel", "pin", "bro", "pur", "whi", "bla", "blu"]

    for y, row in enumerate(state.board.board):
        for x, tile in enumerate(row):
            if tile >= 0:
                print(f"\\draw{cols[tile]}{{({x},{y})}}")
        print()
