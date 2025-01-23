import subprocess
import yaml
import matplotlib.pyplot as plt
import os
from primal_tree import *


def write_data(costs, values):
    costs = np.array(costs)
    values = np.array(values)

    np.savetxt("data/figure2data.txt", np.vstack([costs,values]))


def find_values(path):
    found_gross_gain = False
    found_optimal_gain = False
    found_greedy = False
    with open(path, "r") as file:
        while not (found_optimal_gain and found_gross_gain and found_greedy):
            parts = file.readline().strip().split(":")
            if len(parts) == 2:  # Ensure the line has a key-value structure
                key, value = parts[0].strip(), parts[1].strip()

                if key == "gross_gain":
                    gross_gain = float(value)  # Convert the value to a float
                    found_gross_gain = True
                elif key == "optimal_gain":
                    optimal_gain = float(value)  # Convert the value to a float
                    found_optimal_gain = True
                elif key == "greedy_gain":
                    greedy_gain = float(value)
                    found_greedy=True
        return gross_gain, optimal_gain, greedy_gain


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
            # with open(path+filename+f"{k}.yaml", 'r') as f:
            #     data = yaml.safe_load(f)
            current, optimal, greedy = find_values(path+filename+f"{k}.yaml")
            # optimal = data["nodes"][0]["optimal_gain"]
            # current = data["nodes"][0]["gross_gain"]
            gains.append((current-greedy)/(optimal-greedy))
            os.remove(path+filename+f"{k}.yaml")
        plt.plot(costs, gains)

        allgains.append(gains)
    write_data(costs, allgains)
    plt.show()
