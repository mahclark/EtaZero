# -*- coding: utf-8 -*-
"""ArenaRunner.ipynb
"""

from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)
import sys
base_path = "/content/gdrive/My Drive/EtaZero/src"
sys.path.append(base_path)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from agent_evaluation.arena import Arena
from agents.eta_zero import EtaZero
from agents.random_agent import RandomAgent
from agents.network_agent import RawNetwork
from agents.uct_agent import UCTAgent
from networks.graph_networks import PolicyValRGCN
import datetime
import os
import torch
import utils
section = "Attempt7"

arena = Arena(base_path, section, saving_enabled=False)
arena.plot_all()

arena = Arena(base_path, section=section)
arena.add_task(
    EtaZero.Series(50, base_path, section, lower_limit=70),
    EtaZero.Series(50, base_path, section, lower_limit=70),
    game_pairs=20,
    shift=[1,10]
)
smps = list((5, 10, 20) + UCTAgent.Series.all_samples)
arena.add_task(
    UCTAgent.Series(smps),
    UCTAgent.Series(smps),
    game_pairs=20,
    shift=[1,2,5]
)
smps.remove(1000)
arena.add_task(
    EtaZero.Series(50, base_path, section, lower_limit=114),
    UCTAgent.Series(smps),
    game_pairs=40
)
arena.start()

arena = Arena(base_path, section=section)
arena.add_task(
    EtaZero.Series(50, base_path, section),
    RandomAgent.Series(),
    game_pairs=20
)
arena.start()