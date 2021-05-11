# -*- coding: utf-8 -*-
"""Base-9 Tester.ipynb
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

custom_eta = EtaZero(utils.load_net(114, base_path, section), 50)
custom_eta.elo_id += "-base-9"
def custom_members():
  return [custom_eta]

eta_series = EtaZero.Series(50, base_path, section, lower_limit=70)
eta_series.get_members = custom_members

arena = Arena(base_path, section=section)
arena.add_task(
    eta_series,
    RandomAgent.Series(),
    game_pairs=20,
    base=9
)
arena.add_task(
    eta_series,
    UCTAgent.Series([5000, 10_000]),
    game_pairs=40,
    base=9
)
arena.start()

