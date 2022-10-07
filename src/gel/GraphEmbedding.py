import os
import numpy as np
import pandas as pd

from dynamicgem.embedding.dynAERNN import DynAERNN
from dynamicgem.embedding.ae_static import AE
from dynamicgem.embedding.dynAE import DynAE
from dynamicgem.embedding.dynRNN import DynRNN

from gel import CppWrapper as N2V
from cmn import Common as cmn
import Params

def embedding(dim_emb, lookback, method='DynAERNN', n_users=100):
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
                       n_batch=min(n_users,10),
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
    users = np.load(f"{Params.uml['path2save']}/Users.npy")
    if not os.path.isdir(Params.gel["path2save"]): os.makedirs(Params.gel["path2save"])
    if method.lower() == 'node2vec':
        N2V.main(Params.uml['path2save']+'/graphs', Params.gel['path2save'], Params.gel['embeddingDim'])
    else:
        lookback = 2
        embedding_instance = embedding(dim_emb=Params.gel['embeddingDim'], lookback=lookback, method=method, n_users=len(users))
        if method.lower() == "ae": emb, _ = embedding_instance.learn_embeddings(graphs[-1])
        else: emb, _ = embedding_instance.learn_embeddings(graphs)

        embeddings = {}
        for i, u in enumerate(users):
            embeddings[u] = emb[i]
        pd.to_pickle(embeddings, f'{Params.gel["path2save"]}/Embeddings.pkl')
        return embeddings
