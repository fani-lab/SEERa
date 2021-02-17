import ge #https://github.com/shenweichen/GraphEmbedding
import gem #https://github.com/palash1992/GEM
import ampligraph

from gem.utils      import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr
import networkx as nx
from gem.embedding.gf       import GraphFactorization
from gem.embedding.hope     import HOPE
from gem.embedding.lap      import LaplacianEigenmaps
from gem.embedding.lle      import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne     import SDNE
from time import time
import matplotlib.pyplot as plt


G = nx.random_graphs.random_regular_graph(8,12)
# embedding = HOPE(d=128, beta=0.01)
# t1 = time()
# res = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
# print(embedding._method_name, ':\n\tTraining time: %f' % (time() - t1))


# Drawing
a = nx.graph.Graph([[1,2],[2,3],[4,5],[3,4],[2,5]])
nx.draw_networkx(G, with_labels = True)
plt.interactive(False)
plt.show()
