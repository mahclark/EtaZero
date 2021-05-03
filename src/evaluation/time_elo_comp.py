import matplotlib.pyplot as plt
import os
from evaluation.arena import LockParser
from evaluation.agent_timer import AgentTimer

if __name__ == "__main__":
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(6.3, 4))

    ratings, _ = LockParser.read(
        os.path.join("data", "elo", "Attempt7", "ratings.json")
    )

    timer = AgentTimer()
    timer.sys_id = "colab"
    avg_times = timer.avg_times()

    max_times = {
        time_id: max(t for t in data.values()) for time_id, data in avg_times.items()
    }

    eta_x = []
    eta_y = []
    uct_x = []
    uct_y = []

    for time_id, t in sorted(max_times.items(), key=lambda x: int(x[0].split("-")[1])):
        if time_id in ratings:
            rating = ratings[time_id]
        else:
            try:
                rating = max(
                    rating for eid, rating in ratings.items() if time_id + "-" in eid
                )
            except ValueError:
                continue

        if "EtaZero" in time_id:
            eta_x.append(t)
            eta_y.append(rating)
        else:
            uct_x.append(t)
            uct_y.append(rating)

        if time_id == "EtaZero-50":
            plt.text(x=t - 0.4, y=rating - 70, s=time_id)
        else:
            plt.text(x=t + 0.5, y=rating - 40, s=time_id)

    plt.plot(eta_x, eta_y, linestyle=":", marker="o")
    plt.plot(uct_x, uct_y, linestyle=":", marker="o", color="C2")

    plt.xlim([0, 30])
    plt.xlabel("Maximum Average Move Time (s)")
    plt.ylabel("Elo Rating")
    plt.show()
