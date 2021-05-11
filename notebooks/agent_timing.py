# -*- coding: utf-8 -*-
"""Agent_Timing.ipynb
"""

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
import sys
sys.path.append('/content/gdrive/My Drive/EtaZero')

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline 
import matplotlib.pyplot as plt
import os
import torch
from agent_timer import AgentTimer
from agents.eta_zero import EtaZero
from agents.uct_agent import UCTAgent

timer = AgentTimer("/content/gdrive/My Drive/EtaZero", section="Attempt5")

timer.get_info()

agents = [
  # EtaZero(load_net(1), samples_per_move=200),
  UCTAgent(10_000)
]
for agent in agents:
  timer.time(
      agent,
      num_games=max(30 - timer.get_info().get(agent.time_id, 0), 0)
  )