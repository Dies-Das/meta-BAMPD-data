import json
import pandas as pd
from tqdm import tqdm
import tempfile, os, subprocess


def find_action(current_node, arm):
    result = None
    for action in current_node["actions"]:
        if action["arm"] == arm:
            return action
    return result


def is_computational(current_node):
    computational = False
    for action in current_node["actions"]:
        computational = computational or action["is_computational"]
    return computational


def number_of_computations(current_node):
    computational_actions = 0
    avg = 0
    for action in current_node["actions"]:
        if action["is_computational"]:
            avg += action["computations"]
            computational_actions += 1
    if computational_actions > 0:
        return avg / computational_actions
    else:
        return 0


if __name__ == "__main__":
    datapath = "../data/bandit-stakes-v0.1.1.csv"
    executable = "../bin/meta-BAMDP"
    df = pd.read_csv(datapath)
    arms = 2
    computations = 3
    computational_cost = 0.001
    metapolicies = {}
    subpolicies = {}
    for index, row in tqdm(df.iterrows()):
        if row["trial_idx"] == 0:
            state = [0,0,0,0]
            t = row["horizon"]
            if t not in metapolicies.keys():
                with tempfile.NamedTemporaryFile(
                    dir="/dev/shm", suffix=".json", delete=False
                ) as tempf:
                    path = tempf.name  # e.g., /dev/shm/tmpabcd.csv
                args = [
                    "-o",
                    f"{os.path.dirname(path)}",
                    "-t",
                    f"{t}",
                    f"--max",
                    f"{computational_cost}",
                    f"--filename",
                    os.path.basename(path)[:-4],
                    "-a",
                    f"{arms}",
                    "-n",
                    f"{computations}",
                ]
                cmd = [executable] + args

                subprocess.run(cmd)
                with open(path, "r") as file:
                    metapolicies[t] = json.load(file)
                    metapolicy = metapolicies[t]
                current_node = metapolicy["nodes"]["0"]
                os.remove(path)
            else:
                metapolicy = metapolicies[t]
                current_node = metapolicy["nodes"]["0"]

        computational = is_computational(current_node)
        number = number_of_computations(current_node)
        voc = current_node["voc_bound"]
        greedy_gain = current_node["greedy_gain"]
        optimal_gain = current_node["optimal_gain"]
        df.at[index, "voc_bound"] = voc
        df.at[index, "greedy_gain"] = greedy_gain
        df.at[index, "optimal_gain"] = optimal_gain
        df.at[index, "computational"] = int(computational)
        df.at[index, "number_of_computations"] = number
        if row["horizon"] == row["trial_idx"] + 1:
            continue
        else:
            current_arm = row["arm"]
 
            action = find_action(current_node, current_arm)
            if action == None:
                current_state_list = [current_node["state"][k] for k in range(2 * arms)]
                current_state_list[2 * row["arm"] + (2 * row["reward"]) % 2] += 1
                current_state = ",".join(map(str, current_state_list))
                if tuple((t, current_state)) not in subpolicies.keys():
                    with tempfile.NamedTemporaryFile(
                        dir="/dev/shm", suffix=".json", delete=False
                    ) as tempf:
                        path = tempf.name  # e.g., /dev/shm/tmpabcd.csv
                    args = [
                        "-o",
                        f"{os.path.dirname(path)}",
                        "-t",
                        f"{t}",
                        f"--max",
                        f"{computational_cost}",
                        f"--filename",
                        os.path.basename(path)[:-4],
                        "-a",
                        f"{arms}",
                        "-n",
                        f"{computations}",
                        "--state",
                        f"{current_state}",
                    ]
                    cmd = [executable] + args
                    subprocess.run(cmd)
                    with open(path, "r") as file:
                        subpolicies[tuple((t, current_state))] = json.load(file)
                    current_node = subpolicies[tuple((t, current_state))]["nodes"]["0"]
                    os.remove(path)
                else:
                    current_node = subpolicies[tuple((t, current_state))]["nodes"]["0"]

                metapolicy = subpolicies[tuple((t, current_state))]
            else:

                if row["reward"]:
                    current_node = metapolicy["nodes"][action["children"][0]]
                else:
                    current_node = metapolicy["nodes"][action["children"][1]]

    df.to_csv("../data/test.csv", index=False)
