import os
import networkx as nx
import matplotlib.pyplot as plt
from dynamicgem.embedding.dynAERNN import DynAERNN
from dynamicgem.embedding.ae_static    import AE
from dynamicgem.embedding.dynAE        import DynAE
from dynamicgem.embedding.dynRNN       import DynRNN
import glob
from dynamicgem.graph_generation import dynamic_SBM_graph as sbm
from dynamicgem.visualization import plot_dynamic_sbm_embedding
from time import time
from sklearn.metrics.pairwise import cosine_similarity
# import ToyGraphGeneration as TGG
import numpy as np

def GEMmethod(dim_emb, lookback, method='DynAERNN'):
    methods = ['AE', 'DynAE', 'DynRNN', 'DynAERNN']
    if method == methods[0]:
        embedding = AE(d=dim_emb,
                       beta=5,
                       nu1=1e-6,
                       nu2=1e-6,
                       K=3,
                       n_units=[500, 300, ],
                       n_iter=200,
                       xeta=1e-4,
                       n_batch=100,
                       modelfile=['./GEL/enc_model_AE.json',
                                  './GEL/dec_model_AE.json'],
                       weightfile=['./GEL/enc_weights_AE.hdf5',
                                   './GEL/dec_weights_AE.hdf5'])
    elif method == methods[1]:
        embedding = DynAE(d=dim_emb,
                          beta=5,
                          n_prev_graphs=lookback,
                          nu1=1e-6,
                          nu2=1e-6,
                          n_units=[500, 300, ],
                          rho=0.3,
                          n_iter=250,
                          xeta=1e-4,
                          n_batch=100,
                          modelfile=['./intermediate/enc_model_dynAE.json',
                                     './intermediate/dec_model_dynAE.json'],
                          weightfile=['./intermediate/enc_weights_dynAE.hdf5',
                                      './intermediate/dec_weights_dynAE.hdf5'],
                          savefilesuffix="testing")
    elif method == methods[2]:
        embedding = DynRNN(d=dim_emb,
                           beta=5,
                           n_prev_graphs=lookback,
                           nu1=1e-6,
                           nu2=1e-6,
                           n_enc_units=[500, 300],
                           n_dec_units=[500, 300],
                           rho=0.3,
                           n_iter=250,
                           xeta=1e-3,
                           n_batch=100,
                           modelfile=['./intermediate/enc_model_dynRNN.json',
                                      './intermediate/dec_model_dynRNN.json'],
                           weightfile=['./intermediate/enc_weights_dynRNN.hdf5',
                                       './intermediate/dec_weights_dynRNN.hdf5'],
                           savefilesuffix="testing")
    elif method == methods[3]:
        embedding = DynAERNN(d=dim_emb,
                             beta=5,
                             n_prev_graphs=lookback,
                             nu1=1e-6,
                             nu2=1e-6,
                             n_aeunits=[500, 300],
                             n_lstmunits=[500, dim_emb],
                             rho=0.3,
                             n_iter=200,
                             xeta=1e-3,
                             n_batch=100,
                             modelfile=['./GEL/enc_model_dynAERNN.json',
                                        './GEL/dec_model_dynAERNN.json'],
                             weightfile=['./GEL/enc_weights_dynAERNN.hdf5',
                                         './GEL/dec_weights_dynAERNN.hdf5'],
                             savefilesuffix="testing")
    return embedding
def main(method='DynAERNN'):
    # Parameters for Stochastic block model graph
    # Todal of 1000 nodes
    node_num = 1000
    edge_num = 8000
    # Test with two communities
    community_num = 3
    # At each iteration migrate 10 nodes from one community to the another
    node_change_num = 4
    # Length of total time steps the graph will dynamically change

    # output directory for result
    # outdir = './output'
    # intr = './intermediate'
    # if not os.path.exists(outdir):
    #     os.mkdir(outdir)
    # if not os.path.exists(intr):
    #     os.mkdir(intr)
    testDataType = 'sbm_cd'
    # Generate the dynamic graph
    # dynamic_sbm_series = list(sbm.get_community_diminish_series_v2(node_num,
    #                                                                community_num,
    #                                                                length,
    #                                                                1,  # comminity ID to perturb
    #                                                                node_change_num))
    # dynamic_sbm_series = np.asarray(dynamic_sbm_series)
    # print('dynamic_sbm_series shape: ', dynamic_sbm_series.shape)
    # np.save('dynamic_sbm_series.npy', dynamic_sbm_series)
    graphs1 = []
    # graphs2 = []
    # if not os.path.exists('graphs'):
    #     os.mkdir('graphs')
    # os.chdir('graphs')
    # for i in range(length):
    #     g1, g2 = TGG.randomGraph(n=node_num, e=edge_num)
    #     g = TGG.Test()
    #     nx.write_gpickle(g2, 'g2_'+str(i)+'.net')
    #     # graphs1.append(g1)
    #     graphs2.append(g2)
    # graphs = graphs2
    # graphs = [nx.read_gpickle('scenario3/graph_day1.net'), nx.read_gpickle('scenario3/graph_day2.net'),
    #           nx.read_gpickle('scenario3/graph_day3.net'), nx.read_gpickle('scenario3/graph_day4.net'),
    #           nx.read_gpickle('scenario3/graph_day5.net'), nx.read_gpickle('scenario3/graph_day6.net'),
    #           nx.read_gpickle('scenario3/graph_day7.net')]
    graphs_path = glob.glob('*.net')
    graphs = []
    for gp in graphs_path:
        graphs.append(nx.read_gpickle(gp))
    print('len graph: ', len(graphs))
    # graphs = [nx.read_gpickle('graph_day1.net'), nx.read_gpickle('graph_day2.net'),
    #           nx.read_gpickle('graph_day3.net'), nx.read_gpickle('graph_day4.net'),
    #           nx.read_gpickle('graph_day5.net'), nx.read_gpickle('graph_day6.net'),
    #           nx.read_gpickle('graph_day7.net'), nx.read_gpickle('graph_day8.net'),
    #           nx.read_gpickle('graph_day9.net')]
    length = len(graphs)
    np.save('length.npy', length)
    # os.chdir('scenario3')
    # graphsnew = [g[0] for g in dynamic_sbm_series]
    # print('GRAPHS NEW: ')
    # print('type: ', type(graphsnew))
    # print('subtype: ', type(graphsnew[0]))
    # print('shape: ', np.asarray(graphsnew).shape)
    print('GRAPHS: ')
    print('type: ', type(graphs))
    print('subtype: ', type(graphs[0]))
    print('graphs shape: ', np.asarray(graphs).shape)
    np.save('graphs.npy', graphs)

    # parameters for the dynamic embedding
    # dimension of the embedding
    dim_emb = 100
    lookback = 2
    print('lookback: ', lookback)

    # methods: ['AE', 'DynAE', 'DynRNN', 'DynAERNN'] are available.
    embedding = GEMmethod(dim_emb=dim_emb, lookback=lookback, method=method)

    embs = []
    t1 = time()
    for temp_var in range(lookback + 1, length + 1):
        emb, _ = embedding.learn_embeddings(graphs[:temp_var])
        print('emb type: ', type(emb))
        embs.append(emb)
    embs = np.asarray(embs)
    print('embs shape: ', embs)
    print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))
    plt.figure()
    plt.clf()
    # plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding_v2(embs[-5:-1], dynamic_sbm_series[-5:])
    # plt.savefig('myname.png')
    np.save('embeddeds.npy', embs)
    plt.show()


# if __name__ == '__main__':
#     os.chdir('run_outputs/2021_03_09__02_11/graphs')
#     main()
#     emb = np.load('embeddeds.npy')
#     length = np.load('length.npy')
#     # dss = np.load('dynamic_sbm_series.npy', allow_pickle=True)
#     grp = np.load('graphs.npy')
#     e = emb[-1]
#     # g = nx.read_gpickle('graph_day14.net')
#     usersSimilarity = cosine_similarity(e)
#     print('min of similarities: ', usersSimilarity.min())


os.chdir('run_outputs/2021_03_31__03_24/graphs')
main()
emb = np.load('embeddeds.npy')
length = np.load('length.npy')
    # dss = np.load('dynamic_sbm_series.npy', allow_pickle=True)
grp = np.load('graphs.npy', allow_pickle=True)
e = emb[-1]
    # g = nx.read_gpickle('graph_day14.net')
usersSimilarity = cosine_similarity(e)
print('min of similarities: ', usersSimilarity.min())