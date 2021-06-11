#Temporary file, will be removed on clean-up.

import re
from utilities import clamp

if __name__ == "__main__":

    correct = 0
    trials = 0
    with open('temp.txt') as f:
        lines = [line.rstrip() for line in f]
        for line in lines:
            reg = re.search("output \((.*?),\), got \[(.*?)\]", line)
            if reg:
                trials += 1
                groups = reg.groups()
                if abs(clamp(float(groups[0])) - clamp(float(groups[1]))) < 0.3:
                    correct += 1
    print("Correct: {} out of {}".format(correct, trials))