"""
This script computes the amount of training data in each of the 3 training sections
"""

import csv
from utils import get_training_files

fnames = get_training_files(base_path="src", section="Attempt7")

g1 = 0
g1t = 0
g2 = 0
g2t = 0
g3 = 0
g3t = 0

for i in range(max(fnames) + 1):
    try:
        with open(f"src\\data\\training\\Attempt7\\{fnames[i]}") as f:
            # print(f"{i}\t:\t{sum(1 for row in f)}")
            if i < 60:
                g1 += sum(1 for row in f)
                g1t += 1
            elif i < 80:
                g2 += sum(1 for row in f)
                g2t += 1
            elif i not in [109, 131]:
                g3 += sum(1 for row in f)
                g3t += 1

    except:
        pass

print(g1 / g1t, g2 / g2t, g3 / g3t)
