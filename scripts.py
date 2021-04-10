from subprocess import check_call


def play():
    check_call(["python", "-m", "game.play"], cwd="src")


def train():
    check_call(["python", "-m", "networks.trainer"], cwd="src")


def arena():
    check_call(["python", "-m", "agent_evaluation.arena"], cwd="src")


def timer():
    check_call(["python", "-m", "agent_evaluation.agent_timer"], cwd="src")


def testnet():
    check_call(["python", "-m", "agent_evaluation.network_tester"], cwd="src")

def fmt():
    check_call(["python", "-m", "black", "src"])