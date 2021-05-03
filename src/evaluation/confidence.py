import numpy as np
import math
import matplotlib.pyplot as plt


def get_range(n, x, conf, max_top=False):
    np.random.seed(38596)
    p_samp = np.random.uniform(size=10000)
    prx = p_samp ** x * (1 - p_samp) ** (n - x)
    w = prx / np.sum(prx)

    i = np.argsort(p_samp)
    p_samp = p_samp[i]
    w = w[i]
    F = np.cumsum(w)

    if max_top:
        tail_size = (1 - conf) / 2
        return (p_samp[F < tail_size][-1], 1)

    tail_size = 1 - conf
    if n == x:
        return (p_samp[F < tail_size][-1], 1)
    elif x == 0:
        return (0, p_samp[F > (1 - tail_size)][0])
    else:
        tail_size = (1 - conf) / 2
        return (p_samp[F < tail_size][-1], p_samp[F > (1 - tail_size)][0])


def get_elo_diff(r):
    lo = max(r[0], 0.001)
    hi = min(r[1], 0.999)
    return 400 * math.log10(1 / lo - 1) - 400 * math.log10(1 / hi - 1)


if __name__ == "__main__":
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(6.3, 4))

    n = 40#160
    w = 12#113

    x = []
    y = []
    for conf in np.arange(0, 1, 0.01):
        diff = get_elo_diff(get_range(n, w, conf))
        x.append(conf)
        y.append(diff)

    plt.xlabel("Confidence in Interval")
    plt.ylabel("Elo Rating Interval Size")
    plt.plot(x, y)

    y95 = get_elo_diff(get_range(n, w, 0.95))
    plt.plot([0.95, 0.95], [0, y95], linestyle=":", color="C2")
    plt.plot([0, 0.95], [y95, y95], linestyle=":", color="C2")
    plt.text(0.89, 3, "0.95")
    plt.text(0.02, y95 + 3, f"{y95:.1f}")

    plt.ylim([0, max(y) + 10])
    plt.xlim([0, 1])
    plt.show()

    elo_diff = -(1462.1-1929.8)#(1994.9 - 1564.1)
    def exp_win(diff):
        return 1/(1 + 10**(diff/400))

    print(f"{exp_win(elo_diff):.1%} +- {exp_win(elo_diff + y95/2):.1%}")
    print(f"{exp_win(elo_diff):.1%} +- {exp_win(elo_diff - y95/2):.1%}")
