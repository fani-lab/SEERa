import networkx as nx
import os
import glob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt



def GraphShow(G,day):

    nx.draw(G, with_labels=True)
    plt.interactive(False)
    # plt.show(block=True)
    plt.savefig('Embedding'+str(day+1)+'.jpg')
    plt.close()


os.chdir('run_outputs/2021_03_11 04_34/graphs')

graphs_path = glob.glob('*.net')
graphs = []
for gp in graphs_path:
    graphs.append(nx.read_gpickle(gp))
print(len(graphs))

embeddeds = np.load('embeddeds.npy')
print('shape of embedded graphs:', embeddeds.shape)
en = 1
for e in embeddeds:
    c = cosine_similarity(e)
    super_threshold_indices = c < 0.9
    c[super_threshold_indices] = 0
    G = nx.from_numpy_matrix(c)

    GraphShow(G, en)
    en += 1
