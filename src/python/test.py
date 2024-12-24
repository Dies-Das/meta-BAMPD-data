from meta_tree import *

size = 5
meta_graph = MetaGraph(size)
print(len(meta_graph.nodes))
policy = {}
#meta_policy((0,0,0,0),frozenset((0,0,0,0)), frozenset(),meta_graph,0.1,policy)
for key, item in meta_graph.nodes.items():
    # if key[0][0]==0:
        print(key[0])