import networkx as nx
import sknetwork as skn
import os

scenario = 'scenario' + str(8)
os.chdir(scenario)

louvain = skn.clustering.Louvain()

graph = nx.read_gpickle('graph_day20.net')
adj = nx.adj_matrix(graph)

lbls_louvain = louvain.fit_transform(adj)
