import networkx as nx
import os
import glob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt



def GraphShow(G,day):
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
    plt.savefig('Embedding'+str(day+1)+'.jpg')
    plt.close()


scenario = 'scenario' + str(10)
os.chdir(scenario)

graphs_path = glob.glob('*.net')
graphs = []
for gp in graphs_path:
    graphs.append(nx.read_gpickle(gp))
print(len(graphs))

# graphs = [nx.read_gpickle('graph_day1.net'), nx.read_gpickle('graph_day2.net'),
#           nx.read_gpickle('graph_day3.net'), nx.read_gpickle('graph_day4.net'),
#           nx.read_gpickle('graph_day5.net'), nx.read_gpickle('graph_day6.net'),
#           nx.read_gpickle('graph_day7.net'), nx.read_gpickle('graph_day8.net'),
#           nx.read_gpickle('graph_day9.net')]

embeddeds = np.load('embeddeds.npy')
print('shape of embedded graphs:', embeddeds.shape)
en = 1
node_number = np.load('node_numbers.npy')
for e in embeddeds:
    c = cosine_similarity(e)
    super_threshold_indices = c < 0.8
    c[super_threshold_indices] = 0
    G = nx.from_numpy_matrix(c)

    d = {}
    # range(x) x is the number of graph clusters
    for i in range(len(node_number)):
        for j in range(node_number[i]):
            d[i*node_number[i]+j] = {'color': i+1}
    nx.set_node_attributes(G, d)
    GraphShow(G, en)
    en += 1
