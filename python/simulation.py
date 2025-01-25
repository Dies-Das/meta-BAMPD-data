import subprocess
import json
import matplotlib.pyplot as plt
import os
from primal_tree import *
import numpy as np
import random

def write_data(costs, values, fname):
    costs = np.array(costs)
    values = np.array(values)

    np.savetxt(f"{fname}", np.vstack([costs, values]))
def get_action(node):
    actions = node["actions"]
    return actions[random.randint(0,len(actions)-1)]
def get_reward(action, probabilities):
    return random.random()>probabilities[action["arm"]]
def single_simulation(data, probabilities):
    arms = len(probabilities)
    current_node = data["nodes"]["0"]
    current_key = "0"
    nr_of_computations = 0
    avg_time_computation = 0
    total_reward_gross = 0
    total_reward_net = 0
    for k in range(data["time_horizon"]+1):

        action = get_action(current_node)
        reward = get_reward(action, probabilities)
        total_reward_gross += reward
        total_reward_net += reward
        total_reward_net -= action["cost_of_action"]
        if action["is_computational"]:
            nr_of_computations += 1
            avg_time_computation += k
        if k<data["time_horizon"]-1:
            current_node = data["nodes"][action["children"][(reward+1)%2]]
    if nr_of_computations>0:
        avg_time_computation /= nr_of_computations
    return total_reward_gross, total_reward_net, avg_time_computation, nr_of_computations

if __name__ == "__main__":
    times = [6,9,12]
    computations = 2
    arms = 2
    min_cost = 0
    max_cost = 0.15
    samples = 20
    step = (max_cost-min_cost)/(samples-1)
    costs = [step*k for k in range(samples)]
    nr_of_simulations = 1000000
    path = "temp/"
    executable = "bin/meta-BAMDP"

    
    for t in times:
        filename = f"t_{t}_c"
        args = ["-o", "temp/", "-t", f"{t}", f"--max", f"{max_cost}", f"--min", f"{
            min_cost}", f"--samples", f"{samples}", f"--filename", filename, "-s", "-a", f"{arms}", "-n", f"{computations}"]
        cmd = [executable] + args
        subprocess.run(cmd)

        for k in range(samples):
            print(f"doing sample {k+1} of {samples}")
            with open(path+filename+f"{k}.yaml", 'r') as f:
                data = json.load(f)
            avg_gain_gross = 0
            avg_gain_net = 0
            avg_time_computation = 0
            nr_of_computations = 0
            for j in range(nr_of_simulations):
                result = single_simulation(data, [0.5,0.5])
                avg_gain_gross += result[0]
                avg_gain_net += result[1]
                avg_time_computation += result[2]
                nr_of_computations += result[3]

            avg_gain_gross /= nr_of_simulations
            avg_gain_net /= nr_of_simulations
            avg_time_computation /= nr_of_simulations
            nr_of_computations /= nr_of_simulations
            print("#############")
            print(f"Average gross gain over simulations: {avg_gain_gross}")
            print(f"Average net gain over simulations: {avg_gain_net}")
            print(f"Average time of computation over simulations: {avg_time_computation}")
            print(f"Average nr of computations over simulations: {nr_of_computations}")
            
            os.remove(path+filename+f"{k}.yaml")


    plt.show()
