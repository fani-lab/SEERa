import os
import networkx as nx
import matplotlib.pyplot as plt
'''
from dynamicgem.embedding.dynAERNN import DynAERNN
from dynamicgem.embedding.ae_static    import AE
from dynamicgem.embedding.dynAE        import DynAE
from dynamicgem.embedding.dynRNN       import DynRNN
'''
import glob
from gel import CppWrapper as N2V
import params
from time import time
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
def main(method='Node2Vec'):
    graphs_path = glob.glob(f'{params.uml["path2saveUML"]}/graphs/*.net')
    graphs = []
    for gp in graphs_path:
        graphs.append(nx.read_gpickle(gp))
    print('len graph: ', len(graphs))
    length = len(graphs)
    np.save('length.npy', length)
    print('GRAPHS: ')
    print('type: ', type(graphs))
    print('subtype: ', type(graphs[0]))
    print('graphs shape: ', np.asarray(graphs).shape)
    np.save('graphs.npy', graphs)

    # parameters for the dynamic embedding
    # dimension of the embedding
    dim_emb = params.gel['EmbeddingDim']
    lookback = 2
    print('lookback: ', lookback)

    # methods: ['AE', 'DynAE', 'DynRNN', 'DynAERNN'] are available.
    if method == 'Node2Vec':
        # if not os.path.isdir(f'{path2_save_uml}/graphs'): os.makedirs(f'{path2_save_uml}/graphs')
        if not os.path.isdir(params.gel["path2saveGEL"]): os.makedirs(params.gel["path2saveGEL"])
        N2V.main(params.uml['path2saveUML']+'/graphs', params.gel['path2saveGEL'], params.gel['EmbeddingDim'])
    else:
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
        # print(os.getcwd())
        # print(f'{params.uml["path2saveGEL"]}/embeddeds.npy')
        np.save(f'{params.gel["path2saveGEL"]}/embeddeds.npy', embs)
        plt.show()
