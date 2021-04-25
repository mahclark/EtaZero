import matplotlib.pyplot as plt
import numpy as np
import utils
from game.sevn import State

if __name__ == "__main__":
    s = State.from_str("2/cc-eac/5.bdeca.2aae.2e2.5")
    s = State.from_str("1/-b-cbgda-g/7.7.7.1eegf2.4d2.4b2.7")

    plt.style.use("seaborn")
    plt.figure(figsize=(12, 2))

    ps = []
    vs = []
    for i in range(126):
        p, v = utils.load_net(i, section="Attempt7").evaluate(s)
        ps.append(p.tolist())
        vs.append(v.tolist())

    tile_names = ["blue", "black", "pink"]
    tile_cols = ["C0", "black", "pink"]

    for i, (p, move) in enumerate(zip(np.array(ps).T, s.get_moves())):
        plt.plot(p, label=f"Policy ({tile_names[i]} tile)", color=tile_cols[i])

    plt.plot(vs, label="Value", color="C1")  # , linestyle=(0, (1, 1)))

    plt.xlabel("Iteration")
    plt.ylabel("Network Output Value")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()
