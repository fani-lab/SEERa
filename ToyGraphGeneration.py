import os
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt


Num_users = 100
Num_edges = 300


def randomGraph(n=Num_users, e=Num_edges):
    adjacency_mat = np.random.random((n, n))
    adjacency_mat = (adjacency_mat+adjacency_mat.T)/2
    np.fill_diagonal(adjacency_mat, 1)
    g1 = nx.random_graphs.gnm_random_graph(n, e)
    for (u, v) in g1.edges():
        g1[u][v]['weight'] = round(random.random(), 2)
    g2 = nx.from_numpy_matrix(adjacency_mat)
    return g1, g2
    # return g2


def denseGraph(n):
    return nx.complete_graph(n)


def emptyGraph(n):
    return nx.empty_graph(n)


def GraphsConnection(G_array, rate):
    if len(G_array) < 2:
        print('Array must contain more than one graph')
        return 0
    G = nx.disjoint_union(G_array[0], G_array[1])
    nodesnumber = [x.number_of_nodes() for x in G_array]
    connections = round(min(nodesnumber[0], nodesnumber[1]) * rate)
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
        print((node1, node2))
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


def GraphShow(G, day):
    color_map = []
    for node in range(len(G.nodes)):
        if G.nodes[node]['color'] == 1:
            color_map.append('red')
        elif G.nodes[node]['color'] == 2:
            color_map.append('green')
        elif G.nodes[node]['color'] == 3:
            color_map.append('blue')
        elif G.nodes[node]['color'] == 4:
            color_map.append('black')
        elif G.nodes[node]['color'] == 5:
            color_map.append('yellow')
        elif G.nodes[node]['color'] == 6:
            color_map.append('purple')
    nx.draw(G, node_color=color_map, with_labels=True)
    plt.interactive(False)
    # plt.show(block=True)
    strday = str(day)
    if len(strday) == 1:
        strday = '0'+strday
    plt.savefig('day'+strday+'.jpg')
    plt.close()


def graphCreator(n, flag):
    if flag == 1:
        return nx.complete_graph(n)
    elif flag == 0:
        return nx.empty_graph(n)
    else:
        full_edge = n*(n-1)/2
        e = round(flag*full_edge)
        ne, adj_mat = randomGraph(n, e)
        return ne


def Test():
    scenario = 'scenario' + str(10)
    if not os.path.exists(scenario):
        os.mkdir(scenario)
    os.chdir(scenario)
    # status = [
    #     [1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 0, 0, 0, 1, 1],
    #     [0, 0, 1, 1, 1, 0, 0],
    #     [1, 1, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 0]
    # ]
    # status = [
    #     [0.9, 0.9, 0.9, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #     [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    #     [0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    # ]
    status = [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0.9, 0.9, 0.8, 0.7, 0, 0, 0, 0.4, 0.3, 0.2, 0.1, 0.1, 0.3, 0.2, 0.1, 0.1, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1]
              ]
    node_numbers = [10, 10]
    np.save('node_numbers.npy', node_numbers)
    for day in range(len(status[0])):
        g1 = graphCreator(node_numbers[0], status[0][day])
        nx.set_node_attributes(g1, 1, 'color')
        g2 = graphCreator(node_numbers[1], status[1][day])
        nx.set_node_attributes(g2, 2, 'color')
        # g3 = graphCreator(node_numbers[2], status[2][0])
        # nx.set_node_attributes(g3, 3, 'color')
        g = GraphsConnection([g1, g2], 0)

        strday = str(day + 1)
        if len(strday) == 1:
            strday = '0' + strday
        nx.write_gpickle(g, 'graph_day' + strday + '.net')
        GraphShow(g, day + 1)


    '''
    for day in range(len(status[0])):
        g1 = graphCreator(node_numbers[0], status[0][day])
        nx.set_node_attributes(g1, 1, 'color')
        g2 = graphCreator(node_numbers[1], status[1][day])
        nx.set_node_attributes(g2, 2, 'color')
        g3 = graphCreator(node_numbers[2], status[2][day])
        nx.set_node_attributes(g3, 3, 'color')
        # g4 = graphCreator(node_numbers[3], status[3][day])
        # nx.set_node_attributes(g4, 4, 'color')
        # g5 = graphCreator(node_numbers[4], status[4][day])
        # nx.set_node_attributes(g5, 5, 'color')
        # g6 = graphCreator(node_numbers[5], status[5][day])
        # nx.set_node_attributes(g6, 6, 'color')
        g = GraphsConnection([g1, g2, g3], 0)  #, g4, g5, g6], 0)
        strday = str(day+1)
        if len(strday) == 1:
            strday = '0'+strday
        nx.write_gpickle(g, 'graph_day'+strday+'.net')
        GraphShow(g, day+1)
'''
    return g

g = Test()