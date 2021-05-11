# -*- coding: utf-8 -*-
"""EtaTrainer.ipynb
"""

from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)
import sys
base_path = "/content/gdrive/My Drive/EtaZero/src"
sys.path.append(base_path)

from networks.graph_networks import PolicyValRGCN
from networks.trainer import Trainer
from time import perf_counter
import datetime
import os
import utils
section = "Attempt7"

paths = utils.get_model_files(base_path=base_path, section=section)
if len(paths) > 0:
  max_iter = max(paths.keys())
  print(paths[max_iter])

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# model = PolicyValRGCN(dims=[3,128,128,64,64,32,32,16,2])
trainer = Trainer(base_path=base_path, section=section,
                  load_iter=max_iter)
                  # model=model)
t = perf_counter()

completed_games = trainer.count_games()
c7 = min(completed_games, 200)
c5 = completed_games - c7

trainer.eta_training_loop(1, samples_per_move=200, l2norm=10e-4,
  games_5=800-c5, games_7=200-c7, n_epochs=10, lr=0.0003,
  batch_size=64)#, from_train_file="EtaZero-200-PolicyValRGCN-77-2021-02-18-07-34-24.csv")
trainer.eta_training_loop(5, samples_per_move=200, l2norm=10e-4,
  games_5=800, games_7=200, n_epochs=10, lr=0.0003,
  batch_size=64)#, from_train_file="EtaZero-200-PolicyValRGCN-88-2021-03-09-13-00-56.csv")
print(f"total time: {perf_counter() - t:.1f}s")

