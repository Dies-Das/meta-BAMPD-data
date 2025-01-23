import subprocess
import json
import matplotlib.pyplot as plt
import os
from primal_tree import *


def write_data(costs, values):
    costs = np.array(costs)
    values = np.array(values)

    np.savetxt("data/figure2data.txt", np.vstack([costs,values]))



if __name__ == "__main__":
    times = [6,9,12]
    computations = 1
    arms = 3
    min_cost = 0
    max_cost = 0.15
    samples = 200
    step = (max_cost-min_cost)/(samples-1)
    costs = [step*k for k in range(samples)]
    path = "temp/"
    executable = "bin/meta-BAMDP"
    allgains = []
    for t in times:
        filename = f"t_{t}_c"
        args = ["-o", "temp/", "-a", f"{arms}","-t", f"{t}", f"--max", f"{max_cost}", f"--min", f"{
            min_cost}", f"--samples", f"{samples}", f"--filename", filename, "-s", "-n",f"{computations}"]
        cmd = [executable] + args
        subprocess.run(cmd)
        gains = []
        for k in range(samples):
            with open(path+filename+f"{k}.yaml", 'r') as f:
                data = json.load(f)
            greedy = data["nodes"]["0"]["greedy_gain"]
            optimal = data["nodes"]["0"]["optimal_gain"]
            current = data["nodes"]["0"]["gross_gain"]
            gains.append((current-greedy)/(optimal-greedy))
            os.remove(path+filename+f"{k}.yaml")
        plt.plot(costs, gains)

        allgains.append(gains)
    write_data(costs, allgains)
    plt.show()
