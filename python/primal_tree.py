import numpy as np


class PrimalNode:
    # We store a hashmap of all nodes as a class variable. We can access this from each node and keep track of which states are already known, such that we don't duplicate work if two nodes lead to the same state.
    node_dict = {}
    # We can store all relevant information in a Node, number of successes with each decision for example

    def __init__(self, current_depth, max_depth, state):
        self.current_depth = current_depth
        self.max_depth = max_depth
        self.state = state
        self.estimated_probabilities = self.estimate_probabilities(state)
        self.children = []
        self.leftvalue = 0
        self.rightvalue = 0
        self.value = 0
        self.set_value = False
        if self.current_depth == self.max_depth:
            self.value = self.state[0]+self.state[2]
            self.set_value = True
        PrimalNode.node_dict[state] = self

    # We build the complete tree recursively, storing newly created nodes in a hashmap. Should a node corresponding to a state already exist, we don't create a new node.
    def build_tree(self) -> None:
        if self.current_depth == self.max_depth or len(self.children) == 4:
            return
        for k in range(4):
            newstate = [self.state[j] for j in range(4)]
            newstate[k] += 1
            newstate = tuple(newstate)

            if newstate not in PrimalNode.node_dict:
                node = PrimalNode(self.current_depth+1,
                                  self.max_depth, newstate)
                node.build_tree()
            self.children.append(newstate)

    # We recursively evaluate the value of the subtree stemming from this node

    def eval(self):
        # recursion, going through all the children
        if self.state == (2,1,0,0) and self.max_depth==7:
            pass
        if len(self.children) > 0 and not self.set_value:
            self.leftvalue += self.estimated_probabilities[0] * \
                PrimalNode.node_dict[self.children[0]].eval()
            self.leftvalue += (1-self.estimated_probabilities[0]) * \
                PrimalNode.node_dict[self.children[1]].eval()
            self.rightvalue += self.estimated_probabilities[1] * \
                PrimalNode.node_dict[self.children[2]].eval()
            self.rightvalue += (
                1-self.estimated_probabilities[1])*PrimalNode.node_dict[self.children[3]].eval()
            self.value = np.max([self.leftvalue, self.rightvalue])
            self.set_value = True
            return self.value

        else:
            return self.value

    @staticmethod
    def estimate_probabilities(state: np.ndarray) -> np.ndarray:
        p1 = (state[0]+1)/(np.sum(state[:2])+2)
        p2 = (state[2]+1)/(np.sum(state[2:])+2)
        return np.array((p1, p2))

class GreedyNode:
    # We store a hashmap of all nodes as a class variable. We can access this from each node and keep track of which states are already known, such that we don't duplicate work if two nodes lead to the same state.
    node_dict = {}
    # We can store all relevant information in a Node, number of successes with each decision for example

    def __init__(self, current_depth, max_depth, state):
        self.current_depth = current_depth
        self.max_depth = max_depth
        self.state = state
        self.estimated_probabilities = self.estimate_probabilities(state)
        self.children = []
        self.leftvalue = 0
        self.rightvalue = 0
        self.value = 0
        self.set_value = False
        if self.current_depth == self.max_depth:
            self.value = self.state[0]+self.state[2]
            self.set_value = True
        GreedyNode.node_dict[state] = self

    # We build the complete tree recursively, storing newly created nodes in a hashmap. Should a node corresponding to a state already exist, we don't create a new node.
    def build_tree(self) -> None:
        if self.current_depth == self.max_depth or len(self.children) == 4:
            return
        for k in range(4):
            newstate = [self.state[j] for j in range(4)]
            newstate[k] += 1
            newstate = tuple(newstate)

            if newstate not in GreedyNode.node_dict:
                node = GreedyNode(self.current_depth+1,
                                  self.max_depth, newstate)
                node.build_tree()
            self.children.append(newstate)

    # We recursively evaluate the value of the subtree stemming from this node

    def eval(self):
        # recursion, going through all the children
        if len(self.children) > 0 and not self.set_value:
            self.leftvalue += self.estimated_probabilities[0] * \
                GreedyNode.node_dict[self.children[0]].eval()
            self.leftvalue += (1-self.estimated_probabilities[0]) * \
                GreedyNode.node_dict[self.children[1]].eval()
            self.rightvalue += self.estimated_probabilities[1] * \
                GreedyNode.node_dict[self.children[2]].eval()
            self.rightvalue += (
                1-self.estimated_probabilities[1])*GreedyNode.node_dict[self.children[3]].eval()
            close = 1e-7
            if self.estimated_probabilities[0]>self.estimated_probabilities[1]+close:
                self.value = self.leftvalue
            elif self.estimated_probabilities[0]+close<self.estimated_probabilities[1]:
                self.value = self.rightvalue
            else:
                self.value = 0.5*(self.leftvalue+self.rightvalue)
            if self.estimated_probabilities[0] != self.estimated_probabilities[1] and abs(self.estimated_probabilities[0]-self.estimated_probabilities[1])<1e-9:
                raise ValueError
            self.set_value = True
            return self.value

        else:
            return self.value

    @staticmethod
    def estimate_probabilities(state: np.ndarray) -> np.ndarray:
        p1 = (state[0]+1)/(np.sum(state[:2])+2)
        p2 = (state[2]+1)/(np.sum(state[2:])+2)
        return np.array((p1, p2))
# Printing a pretty version of the tree to see precisely the parent-child relationship and the state in each node
def print_tree(node: PrimalNode, level=0, is_last=True, prefix=""):
    # Determine the symbol to use for the current node (corner or straight)
    connector = "└──" if is_last else "├──"
    # Print the current node with the appropriate prefix and connector
    print(prefix + connector +
          f"({node.current_depth},{node.state[:2]},{node.state[2:]}, {node.value}, {node.estimated_probabilities})")

    # Prepare the new prefix for children
    if is_last:
        new_prefix = prefix + "    "
    else:
        new_prefix = prefix + "│   "

    # Recursively print all children, updating the prefix and identifying the last child
    for i, child in enumerate(node.children):
        is_last_child = (i == len(node.children) - 1)
        print_tree(PrimalNode.node_dict[child],
                   level + 1, is_last_child, new_prefix)


if __name__ == "__main__":
    # Example for printing. In practice, we would, when used standalone, just initialize the root node with the state we desire and let that node create its children on its own.
    # For the meta-tree, we would create a separate class, that then would create the primal tree on its own without our interaction.
    root = PrimalNode(0, 3, (0, 0, 0, 0))
    root.build_tree()
    root.eval()
    print_tree(root)
