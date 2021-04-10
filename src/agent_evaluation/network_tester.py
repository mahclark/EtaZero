import matplotlib.pyplot as plt
import numpy as np
import utils
from game.sevn import State

if __name__ == "__main__":
    s = State.from_str("2/cc-eac/5.bdeca.2aae.2e2.5")

    plt.style.use("seaborn")

    ps = []
    vs = []
    for i in range(max(utils.get_model_files(section="Attempt7"))):
        p, v = utils.load_net(i, section="Attempt7").evaluate(s)
        ps.append(p.tolist())
        vs.append(v.tolist())

    plt.plot(vs, label="Value")

    for p, move in zip(np.array(ps).T, s.get_moves()):
        plt.plot(p, label=str(move))
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()
