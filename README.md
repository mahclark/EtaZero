# EtaZero

This is my dissertation project for my final year studying as an undergraduate.

EtaZero is an AI built to play the game Sevn (variation of the game [Paletto](https://spielstein.com/games/paletto/rules)).
EtaZero uses the AlphaZero algorithm, but not with a residual convolutional NN but instead with a relational graph NN.

Included in this repository is:
* An implementation of the game, including UI
* Some basic agents to play the game
* AlphaZero algorithm
* Relational graph neural network
* Trainer to create data via self-play and train on it
* Arena to compare and rate the performance of agents
* iOS screen parsing to get the state of a game on the iOS app Sevn
* Data: model checkpoints, training data, timing data, Elo ratings

## Installation

1. Install Python 2.7 or 3.5+
2. [Install Poetry](https://python-poetry.org/docs/)
3. Clone this repo
4. Run `poetry install` to install project dependancies

## Access via Colab

Anyone can use the repo from Google Colab, as demonstrated [here](https://colab.research.google.com/drive/1o3F23d7NKGFEoMF_Q3n1WeBG4p3ybAsu?usp=sharing).
