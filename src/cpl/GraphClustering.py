import os, glob, pickle
from os.path import exists
import networkx as nx
import numpy as np
from scipy.special import softmax
import pandas as pd
from collections import Counter
# import sknetwork as skn #cannot be debugged (in debug mode, raise error)
# from sklearn.metrics.pairwise import pairwise_kernels, linear_kernel, cosine_similarity
from scipy import sparse
import torch
import torchmetrics as tm
from tqdm import tqdm
import time

import Params
from cmn import Common as cmn

def user_cluster_relation(lbls):
    user_ids = np.load(f"{Params.uml['path2save']}/Users.npy")
    user2cluster = {}
    cluster2user = {}
    for i in range(len(lbls)):
        user2cluster[user_ids[i]] = lbls[i]
        cluster2user.setdefault(lbls[i], []).append(user_ids[i])
    pd.to_pickle(user2cluster, f'{Params.cpl["path2save"]}/user2cluster.pkl')
    pd.to_pickle(cluster2user, f'{Params.cpl["path2save"]}/cluster2user.pkl')
    pd.DataFrame.from_dict(user2cluster, orient="index").to_csv(f'{Params.cpl["path2save"]}/user2cluster.csv')
    pd.DataFrame.from_dict(cluster2user, orient="index").to_csv(f'{Params.cpl["path2save"]}/cluster2user.csv')
    return user2cluster, cluster2user

def cluster_topic_interest(clusters, user_topic_interests):
    cluster_topics = {}
    for c in clusters:
        cluster_members = clusters[c]
        cluster_topics[c] = np.zeros(Params.tml['numTopics'])
        for user in cluster_members:
            cluster_topics[c] += user_topic_interests[user].values
        cc = Counter(cluster_topics[c])
        topic, count = cc.most_common()[0]
        count_percentage = (count/len(cluster_topics[c]))*100
        cmn.logger.info(f"Cluster {c} has {len(cluster_members)} users. Topic {cluster_topics[c].argmax()+1} is the favorite topic for {count_percentage}% of users.")
        cluster_topics[c] = softmax(cluster_topics[c])
    cluster_topics = pd.DataFrame(cluster_topics).T
    cluster_topics.to_csv(f'{Params.cpl["path2save"]}/ClusterTopic.csv')
    pd.to_pickle(cluster_topics, f'{Params.cpl["path2save"]}/ClusterTopic.pkl')


def main(embeddings, method):
    try: pred_users_similarity = pd.load_pickle(f'{Params.cpl["path2save"]}/pred_users_similarity.pkl')
    except:
        try:
            embedding_array = pd.read_pickle(f'{Params.cpl["path2save"]}/embedding_array.pkl')
            print('embedding array loaded.')
        except:
            if not os.path.isdir(Params.cpl["path2save"]): os.makedirs(Params.cpl["path2save"])
            cmn.logger.info(f'5.1. Inter-User Similarity Prediction ...')
            user_ids = np.load(f"{Params.uml['path2save']}/Users.npy")
            embedding_array = sparse.csr_matrix(embeddings[user_ids[0]])
            for u in user_ids[1:]:
                embedding_array = sparse.vstack((embedding_array, embeddings[u]))
            pd.to_pickle(embedding_array, f'{Params.cpl["path2save"]}/embedding_array.pkl')
        try:
            pred_users_similarity = torch.load(f'{Params.cpl["path2save"]}/pred_users_similarity.pt')
            cmn.logger.info(f'5.1. pred_users_similarity is loaded ...')
        except:
            embedding_array = sparse.coo_matrix(embedding_array)
            values = embedding_array.data
            indices = np.vstack((embedding_array.row, embedding_array.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            embedding_array_torch = torch.sparse.FloatTensor(i, v, torch.Size(embedding_array.shape)).to_dense()
            pred_users_similarity = tm.functional.pairwise_cosine_similarity(embedding_array_torch)
            pred_users_similarity[Params.uml['userSimilarityThreshold'] > pred_users_similarity] = 0
            # pred_users_similarity = pd.DataFrame(pred_users_similarity)
            # pred_users_similarity = np.random.random(pred_users_similarity.shape)
            # pred_users_similarity[pred_users_similarity < 0.99] = 0

            # pred_users_similarity = sparse.csr_matrix(pred_users_similarity, dtype=np.float16)
            cmn.logger.info(f'Memory needed for pred_users_similarity matrix: {pred_users_similarity.nelement() * pred_users_similarity.element_size()} bytes.')
            torch.save(pred_users_similarity, f'{Params.cpl["path2save"]}/pred_users_similarity.pt')
            pd.to_pickle(pred_users_similarity, f'{Params.cpl["path2save"]}/pred_users_similarity.pkl')
    # cmn.logger.info(f'5.2. Future Graph Prediction ...')#potential bottleneck if huge amount of edges! needs large filtering threshold
    # try: g = nx.read_adjlist(f'{Params.cpl["path2save"]}/Graph.adjlist')
    # except:
    #     g = nx.from_scipy_sparse_matrix(pred_users_similarity)
    #     print(type(pred_users_similarity))
    #     g = nx.from_pandas_adjacency(pred_users_similarity)
    #     nx.write_adjlist(g, f'{Params.cpl["path2save"]}/Graph.adjlist')
    # cmn.logger.info(f"(#Nodes/Users, #Edges): ({g.number_of_nodes()}, {g.number_of_edges()})")
    cmn.logger.info(f'5.3. Future Community Prediction ...')

    try: lbls_louvain = np.load(f'{Params.cpl["path2save"]}/PredUserClusters.npy')
    except:
        from sknetwork.clustering import Louvain
        try:louvain = Louvain(resolution=1, n_aggregations=200, shuffle_nodes=True, return_membership=True, return_aggregate=True, verbose=True)
        except:louvain = Louvain(resolution=1, max_agg_iter=200, shuffle_nodes=True, verbose=1)
        adj = pred_users_similarity.detach().numpy()
        np.save(f'{Params.cpl["path2save"]}/adj.npy', adj)
        cmn.logger.info(f"(size) : ({adj.size})")
        cmn.logger.info(f"(#zeros) : ({adj.size - np.count_nonzero(adj)})")
        adj2 = sparse.csr_matrix(adj)
        pd.to_pickle(adj2, f'{Params.cpl["path2save"]}/adj_sparse.pkl')
        lbls_louvain = np.asarray(louvain.fit_transform(adj2))
        np.save(f'{Params.cpl["path2save"]}/PredUserClusters.npy', lbls_louvain)
        pd.DataFrame(lbls_louvain).to_csv(f'{Params.cpl["path2save"]}/PredUserClusters.csv')
    try:
        u2c = pd.read_pickle(f'{Params.cpl["path2save"]}/user2cluster.pkl')
        c2u = pd.read_pickle(f'{Params.cpl["path2save"]}/cluster2user.pkl')
    except:
        u2c, c2u = user_cluster_relation(lbls_louvain)
    cluster_members = []
    for c in c2u:
        if len(c2u[c]) >= Params.cpl['minSize']: cluster_members.append(len(c2u[c]))

    cmn.logger.info(f"(#Future Communities, Communities Sizes) : ({len(cluster_members)}, {cluster_members})")
    cmn.logger.info(f"(#Future Communities with less then {Params.cpl['minSize']} members) : ({len(c2u) - len(cluster_members)})")

    if not exists(f'{Params.cpl["path2save"]}/ClusterTopic.csv'):
        last_UTI = pd.read_pickle(sorted(glob.glob(f'{Params.uml["path2save"]}/Day*UsersTopicInterests.pkl'))[-1])
        cluster_topic_interest(c2u, last_UTI)
    return lbls_louvain

