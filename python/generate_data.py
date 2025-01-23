import subprocess
import json
import matplotlib.pyplot as plt
import os
from primal_tree import *
import numpy as np


def write_data(costs, values, fname):
    costs = np.array(costs)
    values = np.array(values)

    np.savetxt(f"{fname}", np.vstack([costs, values]))


def get_probability_estimate(state):
    result = np.zeros(int(len(state)/2))
    for k in range(len(result)):
        result[k] = (state[2*k]+1)/(state[2*k]+state[2*k+1]+2)
    return result


def traverse_graph(current, result: dict, data, probabilities_fixed):
    gross_fixed = 0
    net_fixed = 0
    avg_time_uniform = 0
    avg_time_fixed = 0

    if current in result.keys():
        return result[current]
    if sum(data["nodes"][current]["state"]) == data["time_horizon"]-1:
        gross_fixed = np.max(probabilities_fixed)
        net_fixed = gross_fixed
    else:
        probabilities = get_probability_estimate(data["nodes"][current]["state"])
        nr_of_computational = 0
        nr_of_actions = len(data["nodes"][current]["actions"])
        current_time =  sum(data["nodes"][current]["state"])
        for action in data["nodes"][current]["actions"]:
            arm = action["arm"]
            cost_of_action = action["cost_of_action"]
            if action["is_computational"]:
                nr_of_computational += 1
            winning = traverse_graph(action["children"][0], result, data, probabilities_fixed)
            losing = traverse_graph(action["children"][1], result, data, probabilities_fixed)
            gross_fixed += probabilities_fixed[arm]*(winning[0]+1)
            gross_fixed += (1-probabilities_fixed[arm])*(losing[0])
            net_fixed += probabilities_fixed[arm]*(winning[1]+1)
            net_fixed += (1-probabilities_fixed[arm])*(losing[1])
            net_fixed -= cost_of_action
            avg_time_uniform += probabilities[arm]*(winning[2])
            avg_time_uniform += (1-probabilities[arm])*(losing[2])
            avg_time_fixed += probabilities_fixed[arm]*(winning[3])
            avg_time_fixed += (1-probabilities_fixed[arm])*(losing[3])
        if nr_of_computational>0:
            avg_time_uniform +=current_time*(nr_of_computational/nr_of_actions)
            avg_time_fixed += current_time*(nr_of_computational/nr_of_actions)
        gross_fixed /= nr_of_actions
        net_fixed /= nr_of_actions
        

    result[current] = gross_fixed, net_fixed, avg_time_uniform, avg_time_fixed,
    return gross_fixed, net_fixed, avg_time_uniform, avg_time_fixed


if __name__ == "__main__":
    times = [6]
    computations = 3
    arms = 3
    min_cost = 0
    max_cost = 0.15
    samples = 100
    step = (max_cost-min_cost)/(samples-1)
    costs = [step*k for k in range(samples)]
    path = "temp/"
    executable = "bin/meta-BAMDP"
    all_gross_fixed = []
    all_net_fixed = []
    all_time_uniform = []
    all_time_fixed = []
    
    for t in times:
        
        GreedyNode.node_dict = {}
        gr = GreedyNode(0, t, (0, 0, 0, 0))
        gr.build_tree()
        gr.eval()
        greedy_gain = gr.value
        filename = f"t_{t}_c"
        args = ["-o", "temp/", "-t", f"{t}", f"--max", f"{max_cost}", f"--min", f"{
            min_cost}", f"--samples", f"{samples}", f"--filename", filename, "-s", "-a", f"{arms}", "-n", f"{computations}"]
        cmd = [executable] + args
        subprocess.run(cmd)
        gross_fixed = []
        net_fixed = []
        average_time_uniform = []
        time_fixed = []
        for k in range(samples):
            print(f"doing sample {k+1} of {samples}")
            with open(path+filename+f"{k}.yaml", 'r') as f:
                data = json.load(f)
            result = traverse_graph("0", {}, data, [0.5 for k in range(data["arms"])])
            gross_fixed.append(result[0]/data["time_horizon"])
            net_fixed.append(result[1]/data["time_horizon"])
            average_time_uniform.append(result[2]/data["time_horizon"])
            time_fixed.append(result[3]/data["time_horizon"])
            os.remove(path+filename+f"{k}.yaml")
        # plt.plot(costs, average_time_uniform)

        all_gross_fixed.append(gross_fixed)
        all_net_fixed.append(net_fixed)
        all_time_uniform.append(average_time_uniform)
        all_time_fixed.append(time_fixed)
        
    write_data(costs, all_gross_fixed, "data/gross_fixed.dat")
    write_data(costs, all_net_fixed, "data/net_fixed.dat")
    write_data(costs, all_time_uniform, "data/time_uniform.dat")
    write_data(costs, all_time_fixed, "data/time_fixed.dat")
    
    plt.show()
