import os
import matplotlib.pyplot as plt
from time import time
import numpy as np

from dynamicgem.embedding.dynAERNN import DynAERNN
from dynamicgem.embedding.ae_static import AE
from dynamicgem.embedding.dynAE import DynAE
from dynamicgem.embedding.dynRNN import DynRNN

from gel import CppWrapper as N2V
import params

def GEMmethod(dim_emb, lookback, method='DynAERNN'):
    methods = ['AE', 'DynAE', 'DynRNN', 'DynAERNN']
    if method == methods[0]:
        embedding = AE(d=dim_emb,
                       beta=5,
                       nu1=1e-6,
                       nu2=1e-6,
                       K=3,
                       n_units=[500, 300, ],
                       n_iter=params.gel['epoch'],
                       xeta=1e-4,
                       n_batch=100,
                       modelfile=['./GEL/enc_model_AE.json', './GEL/dec_model_AE.json'],
                       weightfile=['./GEL/enc_weights_AE.hdf5', './GEL/dec_weights_AE.hdf5'])
    elif method == methods[1]:
        embedding = DynAE(d=dim_emb,
                          beta=5,
                          n_prev_graphs=lookback,
                          nu1=1e-6,
                          nu2=1e-6,
                          n_units=[500, 300, ],
                          rho=0.3,
                          n_iter=params.gel['epoch'],
                          xeta=1e-4,
                          n_batch=100,
                          modelfile=['./intermediate/enc_model_dynAE.json', './intermediate/dec_model_dynAE.json'],
                          weightfile=['./intermediate/enc_weights_dynAE.hdf5', './intermediate/dec_weights_dynAE.hdf5'],
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
                           n_iter=params.gel['epoch'],
                           xeta=1e-3,
                           n_batch=100,
                           modelfile=['./intermediate/enc_model_dynRNN.json', './intermediate/dec_model_dynRNN.json'],
                           weightfile=['./intermediate/enc_weights_dynRNN.hdf5', './intermediate/dec_weights_dynRNN.hdf5'],
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
                             n_iter=params.gel['epoch'],
                             xeta=1e-3,
                             n_batch=100,
                             modelfile=['./GEL/enc_model_dynAERNN.json', './GEL/dec_model_dynAERNN.json'],
                             weightfile=['./GEL/enc_weights_dynAERNN.hdf5', './GEL/dec_weights_dynAERNN.hdf5'],
                             savefilesuffix="testing")
    return embedding
def main(graphs, method='DynAERNN'):

    # parameters for the dynamic embedding
    # dimension of the embedding
    dim_emb = params.gel['EmbeddingDim']
    # methods: ['Node2Vec', 'AE', 'DynAE', 'DynRNN', 'DynAERNN'] are available.
    if not os.path.isdir(params.gel["path2save"]): os.makedirs(params.gel["path2save"])
    if method == 'Node2Vec':
        # if not os.path.isdir(f'{path2_save_uml}/graphs'): os.makedirs(f'{path2_save_uml}/graphs')
        N2V.main(params.uml['path2save']+'/graphs', params.gel['path2save'], params.gel['EmbeddingDim'])
        #return emb
    else:
        lookback = 2
        print('lookback: ', lookback)
        embedding = GEMmethod(dim_emb=dim_emb, lookback=lookback, method=method)

        embs = []
        t1 = time()
        for temp_var in range(lookback + 1, len(graphs) + 1):
            emb, _ = embedding.learn_embeddings(graphs[:temp_var])
            embs.append(emb)
        embs = np.asarray(embs)
        print('embs shape: ', embs)
        print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))
        # plt.figure()
        # plt.clf()
        # plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding_v2(embs[-5:-1], dynamic_sbm_series[-5:])
        np.savez_compressed(f'{params.gel["path2save"]}/embeddings.npz', a=embs)
        # plt.show()
        return embs
