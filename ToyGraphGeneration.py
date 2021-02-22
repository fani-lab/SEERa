import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt


Num_users = 100
Num_edges = 300


def randomGraph(n=Num_users, e=Num_edges):
    adjacency_mat = 5 * np.random.random((n, n)) - 2.5
    adjacency_mat = (adjacency_mat+adjacency_mat.T)/2
    np.fill_diagonal(adjacency_mat, 1)
    # g1 = nx.random_graphs.gnm_random_graph(n, e)
    # for (u, v) in g1.edges():
    #     g1[u][v]['weight'] = random.randint(0, 500)
    g2 = nx.from_numpy_matrix(adjacency_mat)
    # return g1, g2
    return 1, g2

def denseGraph(n):
    return nx.complete_graph(n)

def GraphsConnection(G_array, rate):
    if len(G_array) < 2:
        print('Array must contain more than one graph')
        return 0

    G = nx.disjoint_union(G_array[0], G_array[1])
    nodesnumber = [x.number_of_nodes() for x in G_array]
    connections = round(min(nodesnumber[0],nodesnumber[1]) * rate)
    print(connections)
    G1nodes = list(G_array[0].nodes)
    G1nodesnumber = G_array[0].number_of_nodes()
    G2nodes = [x + G1nodesnumber for x in G_array[1].nodes]
    print(G1nodes,G2nodes)
    for c in range(connections):
        node1 = random.choice(G1nodes)
        node2 = random.choice(G2nodes)
        G1nodes.remove(node1)
        G2nodes.remove(node2)
        print((node1,node2))
        G.add_edge(node1, node2)
    for i in range(len(G_array) - 2):
        connections = round(min(G.number_of_nodes(), nodesnumber[i + 2]) * rate)
        print(connections)
        G1nodes = list(G.nodes)
        G1nodesnumber = G.number_of_nodes()
        G2nodes = [x + G1nodesnumber for x in G_array[i+2].nodes]
        print(G1nodes, G2nodes)
        G = nx.disjoint_union(G, G_array[i + 2])
        for c in range(connections):
            node1 = random.choice(G1nodes)
            node2 = random.choice(G2nodes)
            G1nodes.remove(node1)
            G2nodes.remove(node2)
            print((node1, node2))
            G.add_edge(node1, node2)
    return G



def GraphShow(G):
    nx.draw(G)
    plt.interactive(False)
    plt.show(block=True)


def Test():
    g1 = nx.complete_graph(3)
    g2 = nx.complete_graph(4)
    g3 = nx.complete_graph(5)
    g4 = nx.complete_graph(6)
    g = GraphsConnection([g1, g2, g3, g4], 0.2)
    GraphShow(g)


Test()