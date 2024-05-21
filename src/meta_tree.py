from primal_tree import *
from functools import cache
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt
from math import isclose

expansions = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))



@cache
def subgraph(root, nodes, edges, max_depth):
    """
    Create a subgraph starting at root given the nodes in nodes.

    Parameters:
    root (tuple): A 4-tuple of integers representing the root node of the graph.
    nodes (frozenset): A frozenset of nodes in the graph, where each node is presumably a tuple.
    max_depth (int): The maximum depth of our bandit task.

    Returns:
    frozenset: A set of nodes representing the subgraph.
    """
    if root not in nodes:
        return frozenset([root]), frozenset()
    result_nodes = set([root])
    result_edges = set()
    expand_subgraph(root, nodes, edges, result_nodes, result_edges, max_depth)
    return frozenset(result_nodes), frozenset(result_edges)


def expand_subgraph(root, nodes, edges, result_nodes, result_edges, max_depth):
    if sum(root) == max_depth-1:
        return
    children = get_children(root)
    for child in children:
        # child = to_id(child, max_depth)
        if child in nodes and (root, child) in edges:
            result_nodes.add(child)
            result_edges.add((root, child))
            expand_subgraph(child, nodes, edges, result_nodes,
                            result_edges, max_depth)


@cache
def eval_basegraph(root, nodes, edges, max_depth, special_node=None, special_node_value=None):
    """
    Evaluate the BaseGraph structure given a root node and the set of nodes of the graph.

    This function processes a graph starting from the root node and examines connections up to a specified depth. It calculates and returns an array of two floats representing the left and right expected gain respectively.

    Parameters:
    root (tuple): A 4-tuple of integers representing the root node of the graph.
    nodes (frozenset): A frozenset of nodes in the graph, where each node is presumably a tuple.
    max_depth (int): The maximum depth of our bandit task.

    Returns:
    numpy.array: An array of two float values derived from evaluating the graph up to the specified maximum depth.
    """
    # root = to_state(root, max_depth)
    if root == special_node:
        return special_node_value
    current = sum(root)
    probs = get_probabilities(root)
    # if current==max_depth-1:
    #     return

    children = get_children(root)
    value = np.zeros(2)
    number_of_children = 0

    for k, child in enumerate(children):

        # child = to_id(child, max_depth)

        if child in nodes and (root, child) in edges:
            number_of_children += 1
            sub_graph_nodes, sub_graph_edges = subgraph(
                child, nodes, edges, max_depth)
            if k % 2 == 0:
                value[k//2] += probs[k//2] * \
                    np.max(eval_basegraph(child, sub_graph_nodes,
                           sub_graph_edges, max_depth))
            else:
                value[k//2] += (1-probs[k//2])*np.max(eval_basegraph(child,
                                                                     sub_graph_nodes, sub_graph_edges, max_depth))
        else:
            if k % 2 == 0:
                value[k//2] += probs[k//2] * \
                    ((max_depth-current)*probs[k//2]+root[0]+root[2])
            else:
                value[k//2] += (1-probs[k//2]) * \
                    ((max_depth-current)*probs[k//2]+root[0]+root[2])
    return value


class MetaGraph:

    def __init__(self, max_depth, max_compute=-1, max_size=-1):
        self.btilde_sizes = {}
        self.nodes = {}
        self.max_depth = max_depth
        self.max_compute = max_compute
        self.max_size = max_size

        PrimalNode.node_dict = {}
        primal = PrimalNode(0, max_depth, (0, 0, 0, 0))
        primal.build_tree()
        primal.eval()
        GreedyNode.node_dict = {}
        gr = GreedyNode(0, max_depth, (0, 0, 0, 0))
        gr.build_tree()
        gr.eval()
        self.greedy = {}
        self.optimal = {}
        self.disagreeing_states = set()
        self.max_num_expansions = 0
        self.computational_expansions = 0
        self.s_counter = 0
        close = 1e-8
        self.disagreements = 0
        for state, node in PrimalNode.node_dict.items():
            self.optimal[state] = (node.leftvalue, node.rightvalue)
            self.greedy[state] = (gr.node_dict[state].leftvalue,
                                  gr.node_dict[state].rightvalue, gr.node_dict[state].value)
            probs = get_probabilities(state)
            if node.leftvalue > node.rightvalue+close and probs[1] > probs[0]+close:
                self.disagreeing_states.add(state)
                print(f"Optimal did left, we disagree! at state {state}")
                self.disagreements += 1
            if node.rightvalue > node.leftvalue+close and probs[0] > probs[1]+close:
                print(f"Optimal did right, we disagree! at state {state}")
                self.disagreements += 1
                self.disagreeing_states.add(state)
            if isclose(probs[0], probs[1]) and not isclose(node.rightvalue, node.leftvalue):
                print(
                    f"Optimal did deterministic, we disagree! at state {state}")
                self.disagreements += 1
                self.disagreeing_states.add(state)
        print(f"we have {self.disagreements} disagreements")

        self.root = MetaNode((0, 0, 0, 0), frozenset(
            (0, 0, 0, 0)), frozenset(), self)
        self.nodes[(self.root.state, self.root.nodes,
                    self.root.edges)] = self.root
        self.root.expand()
        print(f"we have changed our mind in {self.s_counter} s states")
        # print(f"did {self.computational_expansions} computational expansion. longest expansion so far is {self.max_num_expansions}, number of metanodes is {len(self.nodes)}")



class MetaNode:

    def __init__(self, state, nodes, edges, tree, parent_terminal=True):
        self.state = state
        self.current_depth = sum(state)
        self.nodes = nodes
        self.edges = edges
        self.computational_length = -1
        self.expanded = False
        self.parent_terminal = parent_terminal
        self.tree: MetaGraph = tree
        if len(edges) > self.tree.max_num_expansions:
            self.tree.max_num_expansions = len(edges)
        self.gains = eval_basegraph(
            self.state, self.nodes, self.edges, self.tree.max_depth)
        self.prefered_action: Action = Action.LEFT
        if self.gains[1] > self.gains[0]:
            self.prefered_action = Action.RIGHT
        self.children = MetaChildren()

    def expand(self):
        # If we already expanded this node, don't try to expand it again. probably unnecessary safety check.
        close = 1e-8
        if self.expanded:
            return
        self.expanded = True
        if sum(self.state) == self.tree.max_depth-1:
            return

        optimal_rewards = self.tree.optimal[self.state]

        potential_children = get_children(self.state)
        children = []
        # If one branch can only be worse then the other, just act
        if optimal_rewards[1]+close < self.gains[0] or (self.state[0] == self.state[2] and self.state[1] == self.state[3]):

            for k in range(0, 2):
                nodes, edges = subgraph(
                    potential_children[k], self.nodes, self.edges, self.tree.max_depth)
                key = (potential_children[k], nodes, edges)

                if key not in self.tree.nodes:

                    self.tree.nodes[key] = MetaNode(
                        potential_children[k], nodes, edges, self.tree)
                    self.tree.nodes[key].expand()
                children.append(MetaChild(*key, 0))

            self.children.left_terminal.append(tuple(children))
            return
        # If one branch can only be worse then the other, just act
        elif optimal_rewards[0]+close < self.gains[1]:
            for k in range(2, 4):
                nodes, edges = subgraph(
                    potential_children[k], self.nodes, self.edges, self.tree.max_depth)
                key = (potential_children[k], nodes, edges)
                if key not in self.tree.nodes:

                    self.tree.nodes[key] = MetaNode(
                        potential_children[k], nodes, edges,  self.tree)
                    self.tree.nodes[key].expand()

                children.append(MetaChild(*key, 0))
            self.children.right_terminal.append(tuple(children))
            return
        # Else, expand along all computational actions
        else:

            if isclose(self.gains[0], self.gains[1]):
                for k in range(2):
                    nodes, edges = subgraph(
                        potential_children[k], self.nodes, self.edges, self.tree.max_depth)
                    key = (potential_children[k], nodes, edges)
                    if key not in self.tree.nodes:

                        self.tree.nodes[key] = MetaNode(
                            potential_children[k], nodes, edges, self.tree)
                        self.tree.nodes[key].expand()
                    children.append(MetaChild(*key, 0))

                self.children.left_terminal.append(tuple(children))

                children = []
                for k in range(2, 4):
                    nodes, edges = subgraph(
                        potential_children[k], self.nodes, self.edges, self.tree.max_depth)
                    key = (potential_children[k], nodes, edges)
                    if key not in self.tree.nodes:

                        self.tree.nodes[key] = MetaNode(
                            potential_children[k], nodes, edges, self.tree)
                        self.tree.nodes[key].expand()
                    children.append(MetaChild(*key, 0))

                self.children.right_terminal.append(tuple(children))
                for action in [Action.LEFT, Action.RIGHT]:
                    self.computational_expansion(
                        self.state, self.nodes, self.edges, action)
            else:
                children = []
                for k in range(2):
                    nodes, edges = subgraph(
                        potential_children[2*self.prefered_action.value+k], self.nodes, self.edges, self.tree.max_depth)
                    key = (
                        potential_children[2*self.prefered_action.value+k], nodes, edges)
                    if key not in self.tree.nodes:

                        self.tree.nodes[key] = MetaNode(
                            potential_children[2*self.prefered_action.value+k], nodes, edges, self.tree)
                        self.tree.nodes[key].expand()
                    children.append(MetaChild(*key, 0))
                if self.prefered_action.value:
                    self.children.right_terminal.append(tuple(children))
                else:
                    self.children.left_terminal.append(tuple(children))
                # self.children[self.prefered_action.value].append(tuple(children))
                # Computational Actions.
                self.computational_expansion(
                    self.state, self.nodes, self.edges, self.prefered_action)

    def computational_expansion(self, root, nodes, edges, prefered_action):

        close = 1e-8
        # Initialize the queue with the root
        queue = deque([(root, nodes, edges)])

        # We will do a priority queue with potential expansion candidates, sorted by the new gain of the new btilde.
        expansion_list = []
        expansion_children = []
        computational_length = -1
        is_disagreeing = True
        checked_pairs = set()
        while queue:

            current_root, current_nodes, current_edges = queue.popleft()
            if (current_root, current_nodes, current_edges) in checked_pairs:
                continue
            else:
                checked_pairs.add((current_root, current_nodes, current_edges))
            if computational_length > -1 and computational_length == 2 and len(current_edges) >= computational_length:
                for val in queue:
                    if len(val[2]) < computational_length and computational_length == 2:
                        print("ignored expansion!")
                break
            if self.tree.max_size != -1 and len(current_edges) > self.tree.max_size*2:
                continue
            if self.tree.max_compute != -1 and len(current_edges)-len(self.edges) > self.tree.max_compute*2:
                continue
            # if current_root != root:
            #     new_gains_bestcase = eval_basegraph(root, current_nodes, current_edges, self.tree.max_depth, current_root, self.tree.optimal[current_root])
            #     new_action_bestcase = Action.LEFT
            #     if new_gains_bestcase[1]>new_gains_bestcase[0]+close:
            #         new_action_bestcase = Action.RIGHT
            #     if new_action_bestcase != prefered_action:
            #         continue
            new_gains = eval_basegraph(
                root, current_nodes, current_edges, self.tree.max_depth)
            if new_gains[(prefered_action.value+1) % 2] > self.tree.optimal[root][prefered_action.value]+close:
                continue
            new_action = Action.LEFT
            if new_gains[1] > new_gains[0]+close:
                new_action = Action.RIGHT

            if new_action != prefered_action:
                computational_length = len(current_edges)
            # Check termination condition
            if sum(current_root) == self.tree.max_depth - 1:
                continue

            optimal_rewards = self.tree.optimal[current_root]
            children = get_children(current_root)
            gain = eval_basegraph(
                current_root, current_nodes, current_edges, self.tree.max_depth)
            expansion = ExpandDirection.BOTH
            is_s = False
            # Decide the direction of expansion based on gain
            if gain[0] > optimal_rewards[1]+close or (current_root == root and prefered_action == Action.RIGHT):
                expansion = ExpandDirection.LEFT
                if current_root != root:
                    is_s = True
            elif gain[1] > optimal_rewards[0]+close or (current_root == root and prefered_action == Action.LEFT):
                expansion = ExpandDirection.RIGHT
                if current_root != root:
                    is_s = True

            # Check left and right children
            for k in [0, 2]:
                if (expansion == ExpandDirection.LEFT and k > 1) or (expansion == ExpandDirection.RIGHT and k < 2):
                    continue

                children_to_explore = [children[k], children[k+1]]

                # Check if the child is already in the btilde, if so, this is a new candidate for our queue
                if (current_root, children[k]) in current_edges:

                    queue.appendleft(
                        (children[k], current_nodes, current_edges))
                    queue.appendleft(
                        (children[k+1], current_nodes, current_edges))

                # Else, create new btildes and add them to the queue if they don't change our mind, if they change our mind this is our maximal number of computations and terminate.
                else:

                    new_nodes_temp = current_nodes.union(children_to_explore)
                    new_edges_temp = current_edges.union(
                        [(current_root, children_to_explore[0]), (current_root, children_to_explore[1])])

                    # queue.appendleft((children[k+1], new_nodes_temp, new_edges_temp))
                    new_gain = eval_basegraph(
                        root, new_nodes_temp, new_edges_temp, self.tree.max_depth)
                    new_prefered = Action.LEFT
                    if new_gain[1] > new_gain[0]:
                        new_prefered = Action.RIGHT
                    if new_prefered != prefered_action:
                        if len(current_edges) == 2 and not is_s:
                            self.tree.s_counter += 1

                        root_children = get_children(self.state)

                        if new_prefered == Action.LEFT:
                            for j in range(2):
                                new_nodes, new_edges = subgraph(
                                    root_children[j], current_nodes, current_edges, self.tree.max_depth)
                                # if we don't have the metanode yet, create it
                                key = (root_children[j], new_nodes, new_edges)
                                self.tree.computational_expansions += 1

                                if self.tree.computational_expansions % 1000 == 0:
                                    print(f"did {self.tree.computational_expansions} computational expansion. longest expansion so far is {
                                          self.tree.max_num_expansions}, number of metanodes is {len(self.tree.nodes)}, current node is {root} with edgelength {len(self.edges)}")

                                if key not in self.tree.nodes:

                                    # if self.tree.computational_expansions%1000==0:
                                    #     print(f"did {self.tree.computational_expansions} computational expansion. longest expansion so far is {self.tree.max_num_expansions}, number of metanodes is {len(self.tree.nodes)}")
                                    self.tree.nodes[key] = MetaNode(
                                        root_children[j], new_nodes, new_edges, self.tree, parent_terminal=False)
                                    self.computational_length = len(new_nodes)
                                    self.tree.nodes[key].expand()

                                expansion_children.append(
                                    MetaChild(*key, len(new_edges_temp)-len(edges)))
                        else:
                            for j in range(2, 4):
                                new_nodes, new_edges = subgraph(
                                    root_children[j], current_nodes, current_edges, self.tree.max_depth)
                                # if we don't have the metanode yet, create it
                                key = (root_children[j], new_nodes, new_edges)
                                self.tree.computational_expansions += 1

                                if self.tree.computational_expansions % 1000 == 0:
                                    print(f"did {self.tree.computational_expansions} computational expansion. longest expansion so far is {
                                          self.tree.max_num_expansions}, number of metanodes is {len(self.tree.nodes)}, current node is {root} with edgelength {len(self.edges)}")

                                if key not in self.tree.nodes:
                                    self.tree.nodes[key] = MetaNode(
                                        root_children[j], new_nodes, new_edges, self.tree, parent_terminal=False)
                                    self.computational_length = len(new_nodes)
                                    expansion_list.append(self.tree.nodes[key])
                                expansion_children.append(
                                    MetaChild(*key, len(new_edges_temp)-len(edges)))

                        if prefered_action.value:
                            self.children.left_computational.append(
                                tuple(expansion_children))
                        else:
                            self.children.right_computational.append(
                                tuple(expansion_children))
                        expansion_children = []
                    else:
                        if is_disagreeing:
                            queue.appendleft(
                                (root, new_nodes_temp, new_edges_temp))


        for item in expansion_list:
            item.expand()


class MetaChild:
    def __init__(self, node, nodes, edges, cost):
        self.node = node
        self.nodes = nodes
        self.edges = edges
        self.cost = cost

    def __call__(self):
        return self.node, self.nodes, self.edges


class MetaChildren:
    def __init__(self):
        self.left_terminal = []
        self.left_computational = []
        self.right_terminal = []
        self.right_computational = []


def meta_policy(node, nodes, edges, graph: MetaGraph, cost, result):
    if (node, nodes, edges) in result:
        return result[node, nodes, edges]
    probabilities = get_probabilities(node)

    if sum(node) == graph.max_depth-1:
        act = Action.BOTH
        if probabilities[0] > probabilities[1]:
            act = Action.LEFT
        elif probabilities[1] > probabilities[0]:
            act = Action.RIGHT
        res = (probabilities[0]+node[0]+node[2], probabilities[1]+node[0]+node[2]
               ), act, (probabilities[0]+node[0]+node[2], probabilities[1]+node[0]+node[2])

        return res

    values_left = [-np.inf]
    values_right = [-np.inf]
    values_left_real = [-np.inf]
    values_right_real = [-np.inf]

    children_left = []
    children_right = []
    value_left_terminal = -np.inf
    value_right_terminal = -np.inf
    value_left_terminal_real = -np.inf
    value_right_terminal_real = -np.inf

    if len(graph.nodes[node, nodes, edges].children.left_terminal) > 0:
        child1 = graph.nodes[node, nodes, edges].children.left_terminal[0][0]
        child2 = graph.nodes[node, nodes, edges].children.left_terminal[0][1]
        value_left_terminal = probabilities[0] * \
            (max(meta_policy(*child1(), graph, cost, result)[0]))
        value_left_terminal += (1-probabilities[0])*(
            max(meta_policy(*child2(), graph, cost, result)[0]))
        value_left_terminal_real = probabilities[0]*(
            real_value(meta_policy(*child1(), graph, cost, result)))
        value_left_terminal_real += (1-probabilities[0])*(
            real_value(meta_policy(*child2(), graph, cost, result)))

    if len(graph.nodes[node, nodes, edges].children.right_terminal) > 0:
        child1 = graph.nodes[node, nodes, edges].children.right_terminal[0][0]
        child2 = graph.nodes[node, nodes, edges].children.right_terminal[0][1]
        value_right_terminal = probabilities[1] * \
            (max(meta_policy(*child1(), graph, cost, result)[0]))
        value_right_terminal += (1-probabilities[1])*(
            max(meta_policy(*child2(), graph, cost, result)[0]))
        value_right_terminal_real = probabilities[1]*(
            real_value(meta_policy(*child1(), graph, cost, result)))
        value_right_terminal_real += (1-probabilities[1])*(
            real_value(meta_policy(*child2(), graph, cost, result)))

    if len(edges) > 0:
        pass
    for k, children in enumerate(graph.nodes[node, nodes, edges].children.left_computational):
        child1: MetaChild = children[0]
        child2: MetaChild = children[1]
        child1_real = meta_policy(*child1(), graph, cost, result)
        child2_real = meta_policy(*child2(), graph, cost, result)


        val1 = probabilities[0]*(max(child1_real[0])-cost*((child1.cost))/2)
        val2 = (1-probabilities[0]) * \
            (max(child2_real[0])-cost*((child2.cost))/2)
        values_left.append(val1+val2)
        val1 = probabilities[0]*(real_value(child1_real))
        val2 = (1-probabilities[0])*(real_value(child2_real))
        values_left_real.append(val1+val2)
        children_left.append((child1, child2))

    for k, children in enumerate(graph.nodes[node, nodes, edges].children.right_computational):

        child1: MetaChild = children[0]
        child2: MetaChild = children[1]
        val3 = probabilities[1]*(max(meta_policy(*child1(),
                                 graph, cost, result)[0])-cost*((child1.cost))/2)
        val4 = (1-probabilities[1])*(max(meta_policy(*child2(),
                                                     graph, cost, result)[0])-cost*((child2.cost))/2)
        values_right.append(val3+val4)
        val3 = probabilities[1] * \
            (real_value(meta_policy(*child1(), graph, cost, result)))
        val4 = (
            1-probabilities[1])*(real_value(meta_policy(*child2(), graph, cost, result)))
        values_right_real.append(val3+val4)
        children_right.append((child1, child2))
    do_computation_left = False
    do_computation_right = False
 
    # Here we check if we're in a state where both immediate terminations are equally likely.
    gains = eval_basegraph(node, nodes, edges, graph.max_depth)
    close = 1e-8

    if isclose(gains[0], gains[1]):
        
        if node[:2] == node[2:4]:
            res = (value_left_terminal, value_left_terminal), Action.BOTH, (value_left_terminal_real, value_left_terminal_real), [
                graph.nodes[node, nodes, edges].children.left_terminal[0]], do_computation_left

        else:
            avg = (value_left_terminal+value_right_terminal)/2
            avg2 = (value_left_terminal_real+value_right_terminal_real)/2

            if max(values_left)-close < avg and max(values_right)-close < avg:
                res = (avg, avg), Action.BOTH, (avg2, avg2), [
                    graph.nodes[node, nodes, edges].children.left_terminal[0], graph.nodes[node, nodes, edges].children.right_terminal[0]], do_computation_left
            elif max(values_left) >= avg+close:
                ind = np.argmax(values_left)
                do_computation_left = True
                res = (max(values_left), value_right_terminal), Action.LEFT, (values_left_real[ind], value_right_terminal_real), [
                    graph.nodes[node, nodes, edges].children.left_computational[ind-1]], do_computation_left
            elif max(values_right) >= avg+close:
                ind = np.argmax(values_right)
                do_computation_right = True
                res = (value_left_terminal, max(values_right)), Action.RIGHT, (value_left_terminal_real, values_right_real[ind]), [
                    graph.nodes[node, nodes, edges].children.right_computational[ind-1]], do_computation_right

    else:

        child_left = None
        child_right = None
        res_left = 0
        res_right = 0
        res_left_real = 0
        res_right_real = 0
        if value_left_terminal > max(values_left)+close:
            child_left = graph.nodes[node, nodes,
                                     edges].children.left_terminal[0]
            res_left = value_left_terminal
            res_left_real = value_left_terminal_real
        else:
            child_id = np.argmax(values_left)
            if child_id != 0:
                child_left = graph.nodes[node, nodes,
                                         edges].children.left_computational[child_id-1]
                do_computation_left = True
            res_left = max(values_left)
            res_left_real = values_left_real[child_id]
        if value_right_terminal > max(values_right)+close:
            child_right = graph.nodes[node, nodes,
                                      edges].children.right_terminal[0]
            res_right = value_right_terminal
            res_right_real = value_right_terminal_real
        else:
            child_id = np.argmax(values_right)
            if child_id != 0:
                child_right = graph.nodes[node, nodes,
                                          edges].children.right_computational[child_id-1]
                do_computation_right = True
            res_right = max(values_right)
            res_right_real = values_right_real[child_id]


        if res_left >= res_right:

            res = (res_left, res_right), Action.LEFT, (res_left_real,
                                                       res_right_real), [child_left], do_computation_left
        else:
            res = (res_left, res_right), Action.RIGHT, (res_left_real,
                                                        res_right_real), [child_right], do_computation_right

    
    result[node, nodes, edges] = res
    return res


def real_value(result):
    if result[1] == Action.LEFT:
        if result[2][0] < 0:
            pass
        return result[2][0]
    else:
        if result[2][1] < 0:
            pass
        return result[2][1]


class ExpandDirection(Enum):
    LEFT = 0
    RIGHT = 1
    BOTH = 2


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    BOTH = 2



@cache
def get_probabilities(state):
    p1 = (state[0]+1.0)/(state[0]+state[1]+2)
    p2 = (state[2]+1.0)/(state[2]+state[3]+2)
    return p1, p2


@cache
def get_children(state):
    return ([tuple(state[k]+expansions[j][k] for k in range(4))for j in range(4)])




@cache
def sum(state):
    return np.sum(state)

def average_computation(state,policy, max_depth):
    if np.sum(state[0]) == max_depth-1:
        return 0
    state_policy = policy[*state]
    val = 0
    if state_policy[-1]:
        val = np.sum(state[0])
    probs = get_probabilities(state[0])
    children = state_policy[3]
    if state_policy[1] == Action.LEFT or state[0][:2] == state[0][2:]:
        return val +probs[0]*average_computation(children[0][0](), policy, max_depth)+(1-probs[0])*average_computation(children[0][1](), policy, max_depth)
    elif state_policy[1] == Action.RIGHT:
        return val +probs[1]*average_computation(children[0][0](), policy, max_depth)+(1-probs[1])*average_computation(children[0][1](), policy, max_depth)
    else:
        
        val += 0.5*probs[0]*average_computation(children[0][0](), policy, max_depth)+(1-probs[0])*average_computation(children[0][1](), policy, max_depth)
        val += 0.5*probs[1]*average_computation(children[1][0](), policy, max_depth)+(1-probs[1])*average_computation(children[1][1](), policy, max_depth)
        return val
def generate_plots_maximum_beliefs(sizes):
    costs = np.linspace(0,0.09,1000)
    meta_policy_values = {(s,k): [] for s in sizes for k in [4,8,12]}
    meta_policy_first_computation =  {(s,k): [] for s in sizes for k in [4,8,12]}
    for k in [4,8,12]:
        eval_basegraph.cache_clear()
        subgraph.cache_clear()
        print(f"doing step {k}")
        
        greedy_values = []
        optimal_values = []
        
        
        for s in sizes:
            meta_graph = MetaGraph(k, max_size = s)
            greedy_value = (meta_graph.greedy[(0,0,0,0)][2])
            optimal_value = max(meta_graph.optimal[(0,0,0,0)])
    
            for cost in costs:
                policy = {}
                meta_policy((0,0,0,0),frozenset((0,0,0,0)), frozenset(),meta_graph,cost,policy)
                meta_policy_first_computation[(s,k)].append(average_computation(((0,0,0,0),frozenset((0,0,0,0)), frozenset()), policy,k))
                meta_policy_value = max(policy[(0,0,0,0),frozenset((0,0,0,0)), frozenset()][2])
                
                meta_policy_values[(s,k)].append((meta_policy_value-greedy_value)/(optimal_value-greedy_value))

        
        
        
        # plt.plot(costs, meta_policy_values_3, label=f"Depth {k}, max exp 5")
    for k in [4,8,12]:
        for s in sizes:
            plt.plot(costs, meta_policy_first_computation[(s,k)], label=f"Average first computation time {s}, {k}")
    plt.legend(loc="upper left")
    
    plt.savefig("plots/first_computation_max_belief.png")
    plt.clf()
    greedy_values = np.zeros(len(costs))
    optimal_values = np.ones(len(costs))
    for k in [4,8,12]:
        for s in sizes:
            plt.plot(costs, meta_policy_values[(s,k)], label=f"Meta Policy Value {s}, {k}")
    plt.legend(loc="lower left")
    
    
    plt.plot(costs, greedy_values)
    plt.plot(costs, optimal_values)
    plt.savefig("plots/meta_value_max_belief.png")

def generate_plots_maximum_expansions(sizes):
    costs = np.linspace(0,0.09,1000)
    meta_policy_values = {(s,k): [] for s in sizes for k in [4,8,12]}
    meta_policy_first_computation =  {(s,k): [] for s in sizes for k in [4,8,12]}
    for k in [4,8,12]:
        eval_basegraph.cache_clear()
        subgraph.cache_clear()
        print(f"doing step {k}")
        
        greedy_values = []
        optimal_values = []
        
        
        for s in sizes:
            meta_graph = MetaGraph(k, max_compute=s)
            greedy_value = (meta_graph.greedy[(0,0,0,0)][2])
            optimal_value = max(meta_graph.optimal[(0,0,0,0)])
    
            for cost in costs:
                policy = {}
                meta_policy((0,0,0,0),frozenset((0,0,0,0)), frozenset(),meta_graph,cost,policy)
                meta_policy_first_computation[(s,k)].append(average_computation(((0,0,0,0),frozenset((0,0,0,0)), frozenset()), policy,k))
                meta_policy_value = max(policy[(0,0,0,0),frozenset((0,0,0,0)), frozenset()][2])
                
                meta_policy_values[(s,k)].append((meta_policy_value-greedy_value)/(optimal_value-greedy_value))

        
        
        
        # plt.plot(costs, meta_policy_values_3, label=f"Depth {k}, max exp 5")
    for k in [4,8,12]:
        for s in sizes:
            plt.plot(costs, meta_policy_first_computation[(s,k)], label=f"Average first computation time {s}, {k}")
    plt.legend(loc="upper left")
    
    plt.savefig("plots/first_computation_max_expansion.png")
    plt.clf()
    greedy_values = np.zeros(len(costs))
    optimal_values = np.ones(len(costs))
    for k in [4,8,12]:
        for s in sizes:
            plt.plot(costs, meta_policy_values[(s,k)], label=f"Meta Policy Value {s}, {k}")
    plt.legend(loc="lower left")
    
    
    plt.plot(costs, greedy_values)
    plt.plot(costs, optimal_values)
    plt.savefig("plots/meta_value_max_expansion.png")
