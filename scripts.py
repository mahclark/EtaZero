from subprocess import check_call


def main():
    check_call(["python", "main.py"], cwd="src")


def arena():
    check_call(["python", "-m", "agent_evaluation.arena"], cwd="src")
