import csv
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    import os

    # print(os.system("dir"))
    plt.style.use("seaborn")

    data = np.genfromtxt("src/evaluation/node_times.csv", dtype=float, delimiter=",")#, names=True)
    rem_tiles, duration, n_eval, n_expand = data[1:,[1,2,4,5]].T

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Remaining Tiles")
    ax1.set_ylabel("Number of Nodes")
    ax1.plot(rem_tiles, n_eval, label="Nodes Evaluated")
    ax1.plot(rem_tiles, n_expand, label="Nodes Expanded")
    ax1.plot(rem_tiles, duration*100, label="Move Time")
    ax1.legend()
    ax1.set_ylim([0,1100])

    ax2 = ax1.twinx()

    ax2.set_ylabel("Duration (s)")
    ax2.grid(False)
    ax2.legend()
    ax2.set_ylim([0,11])

    fig.tight_layout()
    plt.gca().invert_xaxis()
    plt.show()

    # print(a)

    # with open("node_times.csv") as f:
    #     reader = csv.reader(f)

    #     for i, (_,rem,dur,_,n_eval,n_expand,_,_) in f:
    #         if i == 0:
    #             continue
