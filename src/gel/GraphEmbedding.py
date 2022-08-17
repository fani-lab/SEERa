import os
import matplotlib.pyplot as plt
from time import time
import numpy as np
import pickle

from dynamicgem.embedding.dynAERNN import DynAERNN
from dynamicgem.embedding.ae_static import AE
from dynamicgem.embedding.dynAE import DynAE
from dynamicgem.embedding.dynRNN import DynRNN

from gel import CppWrapper as N2V
from cmn import Common as cmn
import Params

def embedding(dim_emb, lookback, method='DynAERNN'):
    # methods = ['AE', 'DynAE', 'DynRNN', 'DynAERNN']
    if method.lower() == 'ae':
        embedding_ = AE(d=dim_emb,
                       beta=5,
                       nu1=1e-6,
                       nu2=1e-6,
                       K=3,
                       n_units=[500, 300, ],
                       n_iter=Params.gel['epoch'],
                       xeta=1e-4,
                       n_batch=100,
                       modelfile=[f'{Params.gel["path2save"]}/enc_model_AE.json', f'{Params.gel["path2save"]}/dec_model_AE.json'],
                       weightfile=[f'{Params.gel["path2save"]}/enc_weights_AE.hdf5', f'{Params.gel["path2save"]}/dec_weights_AE.hdf5'])
    elif method.lower() == 'dynae':
        embedding_ = DynAE(d=dim_emb,
                          beta=5,
                          n_prev_graphs=lookback,
                          nu1=1e-6,
                          nu2=1e-6,
                          n_units=[500, 300, ],
                          rho=0.3,
                          n_iter=Params.gel['epoch'],
                          xeta=1e-4,
                          n_batch=100,
                          modelfile=[f'{Params.gel["path2save"]}/enc_model_dynAE.json', f'{Params.gel["path2save"]}/dec_model_dynAE.json'],
                          weightfile=[f'{Params.gel["path2save"]}/enc_weights_dynAE.hdf5', f'{Params.gel["path2save"]}/dec_weights_dynAE.hdf5'],
                          savefilesuffix="testing")
    elif method.lower() == 'dynrnn':
        embedding_ = DynRNN(d=dim_emb,
                           beta=5,
                           n_prev_graphs=lookback,
                           nu1=1e-6,
                           nu2=1e-6,
                           n_enc_units=[500, 300],
                           n_dec_units=[500, 300],
                           rho=0.3,
                           n_iter=Params.gel['epoch'],
                           xeta=1e-3,
                           n_batch=100,
                           modelfile=[f'{Params.gel["path2save"]}/enc_model_dynRNN.json', f'{Params.gel["path2save"]}/dec_model_dynRNN.json'],
                           weightfile=[f'{Params.gel["path2save"]}/enc_weights_dynRNN.hdf5', f'{Params.gel["path2save"]}/dec_weights_dynRNN.hdf5'],
                           savefilesuffix="testing")
    elif method.lower() == 'dynaernn':
        embedding_ = DynAERNN(d=dim_emb,
                             beta=5,
                             n_prev_graphs=lookback,
                             nu1=1e-6,
                             nu2=1e-6,
                             n_aeunits=[500, 300],
                             n_lstmunits=[500, dim_emb],
                             rho=0.3,
                             n_iter=Params.gel['epoch'],
                             xeta=1e-3,
                             n_batch=100,
                             modelfile=[f'{Params.gel["path2save"]}/enc_model_dynAERNN.json', f'{Params.gel["path2save"]}/dec_model_dynAERNN.json'],
                             weightfile=[f'{Params.gel["path2save"]}/enc_weights_dynAERNN.hdf5', f'{Params.gel["path2save"]}/dec_weights_dynAERNN.hdf5'],
                             savefilesuffix="testing")
    else:
        raise ValueError('Incorrect graph embedding method!')
    return embedding_
def main(graphs, method='DynAERNN'):
    # parameters for the dynamic embedding
    # dimension of the embedding
    dim_emb = Params.gel['embeddingDim']
    # methods: ['Node2Vec', 'AE', 'DynAE', 'DynRNN', 'DynAERNN'] are available.
    if not os.path.isdir(Params.gel["path2save"]): os.makedirs(Params.gel["path2save"])
    if method == 'Node2Vec':
        # if not os.path.isdir(f'{path2_save_uml}/graphs'): os.makedirs(f'{path2_save_uml}/graphs')
        N2V.main(Params.uml['path2save']+'/graphs', Params.gel['path2save'], Params.gel['embeddingDim'])
    else:
        lookback = 2
        embedding_instance = embedding(dim_emb=dim_emb, lookback=lookback, method=method)
        embs = []

        # Using the last graph for AE and all graphs for other baselines (but just one time) #

        # for temp_var in range(lookback + 1, len(graphs) + 1):
        #     if method == "AE":
        #         emb, _ = embedding_instance.learn_embeddings(graphs[temp_var])
        #     else:
        #         emb, _ = embedding_instance.learn_embeddings(graphs[:temp_var])
        #     embs.append(emb)
        # embs = np.asarray(embs)

        if method == "AE": emb, _ = embedding_instance.learn_embeddings(graphs[-1])
        else: emb, _ = embedding_instance.learn_embeddings(graphs)
        embs = [emb]

        # plt.figure()
        # plt.clf()
        # plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding_v2(embs[-5:-1], dynamic_sbm_series[-5:])
        #np.savez_compressed(f'{Params.gel["path2save"]}/embeddings.npz', a=embs)
        with open(f'{Params.gel["path2save"]}/Embeddings.pkl', 'wb') as f: pickle.dump(embs, f)
        # plt.show()
        return np.asarray(embs)
