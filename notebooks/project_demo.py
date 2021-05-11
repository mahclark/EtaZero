# -*- coding: utf-8 -*-
"""Project_Demo.ipynb

# EtaZero Demo
This notebook is a demonstration of my dissertation project and its reproducibility. EtaZero is an AlphaZero-style AI trained to play the game [Sevn](https://apps.apple.com/us/app/sevn/id1179097907).

The project repository contains an implementation of the game, EtaZero's algorithm, trained neural network instances (which are used in the algorithm) and evaluation tools.

First let's clone the project and install some dependancies.
"""

!pip install dgl
!pip install pygame
!pip install portalocker
!git clone https://github.com/mahclark/EtaZero.git

import sys
base_path = "EtaZero/src"
section = "Attempt7"
sys.path.append(base_path)

from evaluation.arena import Arena
from agents.eta_zero import EtaZero
from agents.network_agent import RawNetwork
from agents.random_agent import RandomAgent
from utils import load_net

"""Let's show we have access to the repo by plotting the progress chart of EtaZero. Each point on the line shows the Elo rating of a trained instance of EtaZero."""

arena = Arena(base_path, section)
arena.plot_all(custom_ids=[])

"""Now let's evaluate a new agent: *RawNetwork*

This agent uses a neural network to make decisions immediately without tree-search.

We load a pre-trained neural network and instantiate the agent.
"""

net = load_net(114, base_path, section)
raw_network = RawNetwork(net)

"""Now let's battle our RawNetwork against an agent which randomly selects moves. If the neural network we chose is any good, it should beat the random agent almost all the time."""

arena.battle(
    raw_network,
    RandomAgent(),
    game_pairs=20
)

"""Now let's battle our RawNetwork against an EtaZero instance. Because EtaZero uses the neural network in conjunction with tree search, we expect the RawNetwork to lose most games.

Since EtaZero requires time to search the tree, this battle takes slightly longer.
"""

arena.battle(
    raw_network,
    EtaZero(net, samples_per_move=50),
    game_pairs=5
)

"""Now that it has played some games, we can see an Elo rating was automatically calculated. To compare the rating to other agents, let's replot the progress graph, specifying that we want to include the RawNetwork."""

arena.plot_all(custom_ids=[raw_network.elo_id])