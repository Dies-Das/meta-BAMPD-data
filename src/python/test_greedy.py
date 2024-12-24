from primal_tree import GreedyNode

import time
for arms in [10,20,30,40,50]:
    print("##################")
    start = time.time()
    gr = GreedyNode(0,arms,(0,0,0,0))
    gr.build_tree()
    gr.eval()

    end = time.time()
    
    print(f"Root values: {gr.node_dict[(0,0,0,0)].leftvalue}, {gr.node_dict[(0,0,0,0)].rightvalue}")
    GreedyNode.node_dict = {}
    print(f"number of arms: {arms}")
    print("Execution time (seconds):", end - start)