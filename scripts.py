import sys
from subprocess import check_call


def fmt():
    check_call([sys.executable, "-m", "black", "src", "scripts.py"])


def play():
    check_call([sys.executable, "-m", "game.play"], cwd="src")


def train():
    check_call([sys.executable, "-m", "networks.trainer"], cwd="src")


def arena():
    check_call([sys.executable, "-m", "evaluation.arena"], cwd="src")


def timer():
    check_call([sys.executable, "-m", "evaluation.agent_timer"], cwd="src")


def testnet():
    check_call([sys.executable, "-m", "evaluation.network_tester"], cwd="src")


def minimax():
    check_call([sys.executable, "-m", "evaluation.minimax_tester"], cwd="src")


def timeelo():
    check_call([sys.executable, "-m", "evaluation.time_elo_comp"], cwd="src")
