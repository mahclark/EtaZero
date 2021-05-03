import csv
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(6.3, 2.5))

    data = np.genfromtxt("src/evaluation/node_times.csv", dtype=float, delimiter=",")
    rem_tiles, duration, n_eval, n_expand = data[1:, [1, 2, 4, 5]].T

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Remaining Tiles")
    ax1.set_ylabel("Number of Nodes")
    ax1.plot(rem_tiles, duration * 100, label="Move Time")
    ax1.plot(rem_tiles, n_eval, label="Nodes Evaluated", color="C2")
    ax1.plot(rem_tiles, n_expand, label="Nodes Expanded", color="C3")
    ax1.legend()
    ax1.set_ylim([0, 1100])

    ax2 = ax1.twinx()
    ax2.set_xlim([0, 50])

    ax2.set_ylabel("Duration (s)")
    ax2.grid(False)
    ax2.legend()
    ax2.set_ylim([0, 11])

    fig.tight_layout()
    plt.gca().invert_xaxis()
    plt.show()
