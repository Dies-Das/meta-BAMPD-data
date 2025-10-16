from meta_tree import *
import time
size = 12
start_time = time.time()
meta_graph = MetaGraph(size, 3)
print(f"python took {time.time()-start_time} for metagraph")
print(len(meta_graph.nodes))
print(meta_graph.greedy[(0,0,0,0)])
policy = {}
meta_policy((0,0,0,0),frozenset((0,0,0,0)), frozenset(),meta_graph,0.1,policy)
print(policy[(0,0,0,0),frozenset((0,0,0,0)), frozenset()])
# for key, item in meta_graph.nodes.items():
#     # if key[0][0]==0:
#         print(key)