import os, glob, pickle
import networkx as nx
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import sknetwork as skn
from sklearn.metrics.pairwise import cosine_similarity, pairwise_kernels

import params
from cmn import Common as cmn

def graph_show(g,day):
    g = g.subgraph(list(g.nodes)[:500])
    nx.draw(g)#, with_labels=True)
    plt.interactive(False)
    # plt.show(block=True)
    plt.savefig('Graph'+str(day)+'.jpg')
    plt.close()

def cluster_topic_interest(clusters, user_topic_interests):
    cluster_interests = []
    for i in range(clusters.max()+1):
        cluster_interests.append([])
    for u in range(len(user_topic_interests)):
        cluster = clusters[u]
        cluster_interests[cluster].append(user_topic_interests[u].argmax())
    for ci in range(len(cluster_interests)):
        c = Counter(cluster_interests[ci])
        topic, count = c.most_common()[0]
        count_percentage = (count/len(cluster_interests[ci]))*100
        cmn.logger.info("Cluster "+str(ci)+" has "+str(len(cluster_interests[ci]))+' users. Topic '+str(topic)+' is the favorite topic for '+str(count_percentage)+ '% of users.')


def main(embeddings, path2save, method='louvain', temporal=False):
    if not os.path.isdir(params.cpl["path2save"]): os.makedirs(params.cpl["path2save"])
    cmn.logger.info(f'5.1. Inter-User Similarity Prediction ...')
    user_similarity_threshold = params.uml['userSimilarityThreshold']
    pred_users_similarity = pairwise_kernels(embeddings[-1, :, :], metric='cosine', n_jobs=9)
    pred_users_similarity[pred_users_similarity < user_similarity_threshold] = 0
    # pred_users_similarity = np.random.random(pred_users_similarity.shape)
    # pred_users_similarity[pred_users_similarity < 0.99] = 0
    pred_users_similarity = sparse.csr_matrix(pred_users_similarity)

    cmn.logger.info(f'5.2. Future Graph Prediction ...')#potential bottleneck if huge amount of edges! needs large filtering threshold
    g = nx.from_scipy_sparse_matrix(pred_users_similarity, parallel_edges=False, create_using=None, edge_attribute='weight')
    nx.write_gpickle(g, f'{params.cpl["path2save"]}/Graph.net')
    with open(f'{params.cpl["path2save"]}/Graph.pkl', 'wb') as f: pickle.dump(g, f)

    cmn.logger.info(f'5.3. Future Community Prediction ...')
    cmn.logger.info(f"#Nodes(Users): {g.number_of_nodes()}, #Edges: {g.number_of_edges()}")
    # if method == 'louvain':
    louvain = skn.clustering.Louvain(resolution=1, n_aggregations=200, shuffle_nodes=True, return_membership=True, return_aggregate=True, verbose=1)
    adj = nx.adjacency_matrix(g)
    lbls_louvain = louvain.fit_transform(adj)
    lbls_louvain = np.asarray(lbls_louvain)

    # elif method == 'temporal_louvain':
    #     igraphs = []
    #     adjs = []
    #     for g in graphs:
    #         G = ig.Graph.from_networkx(g)
    #         G.vs['id'] = list(g.nodes())
    #         #ig.plot(G)
    #         igraphs.append(G)
    #         adj = nx.adj_matrix(g)
    #         adjs.append(adj)
    #     print('Louvain2')
    #     print(adj.min(), adj.max(), adj.mean())
    #     lbls_louvain, improvement = lg.find_partition_temporal(igraphs, lg.ModularityVertexPartition, interslice_weight=1)
    #     lbls_louvain = np.asarray(lbls_louvain[-1])

    cluster_members = []
    for UC in range(lbls_louvain.min(), lbls_louvain.max() + 1):
        Users_in_cluster = np.where(lbls_louvain == UC)[0]
        if len(Users_in_cluster) == 1: break
        else: cluster_members.append(len(Users_in_cluster))

    cmn.logger.info(f"#Predicted Future Communities (Louvain): {lbls_louvain.max()}; ({lbls_louvain.max() - len(cluster_members)}) are singleton.")
    cmn.logger.info(f'Communities Size: {cluster_members}')
    np.save(f'{params.cpl["path2save"]}/PredUserClusters.npy', lbls_louvain)
    np.savetxt(f'{params.cpl["path2save"]}/PredUserClusters.csv', lbls_louvain, fmt='%s')

    last_day_UTI = sorted(glob.glob(f'{params.uml["path2save"]}/Day*UsersTopicInterests.npy'))[-1]
    last_UTI = np.load(last_day_UTI)
    cluster_topic_interest(lbls_louvain, last_UTI)
    return lbls_louvain

