import os, glob, pickle
import networkx as nx
from scipy import sparse
import numpy as np
from collections import Counter
import sknetwork as skn
import pandas as pd
from sklearn.metrics.pairwise import pairwise_kernels

import params
from cmn import Common as cmn


def main2(user_features, predicted_features):
    if not os.path.isdir(params.cpl["path2save"]): os.makedirs(params.cpl["path2save"])
    if params.cpl['type'] == 'matrix_based':
        if params.cpl['method'] == 'DBSCAN':
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(predicted_features)
            user_clusters = pd.DataFrame({'UserId': user_features['UserId'], 'Community': labels.tolist()})
            with open(f"{params.cpl['path2save']}/PredUserClusters.npy", 'wb') as f:
                pickle.dump(labels, f)
            user_clusters.to_csv(f'{params.cpl["path2save"]}/user_clusters.csv')
            return user_clusters, labels
        elif params.cpl['method'] == 'AgglomerativeClustering':
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(predicted_features, predicted_features)
            agglomerative = AgglomerativeClustering(metric='precomputed', linkage='average')
            labels = agglomerative.fit_predict(similarities)
            agglomerative2 = AgglomerativeClustering(metric='cosine', linkage='average')
            labels2 = agglomerative2.fit_predict(similarities)
            agglomerative3 = AgglomerativeClustering(metric='l2', linkage='average')
            labels3 = agglomerative3.fit_predict(similarities)
            # assert labels==labels2

            user_clusters = pd.DataFrame({'UserId': user_features['UserId'], 'Community': labels.tolist()})
            with open(f"{params.cpl['path2save']}/PredUserClusters.npy", 'wb') as f:
                pickle.dump(labels, f)
            user_clusters.to_csv(f'{params.cpl["path2save"]}/user_clusters.csv')

            user_clusters2 = pd.DataFrame({'UserId': user_features['UserId'], 'Community': labels2.tolist()})
            with open(f"{params.cpl['path2save']}/PredUserClusters2.npy", 'wb') as f:
                pickle.dump(labels2, f)
            user_clusters2.to_csv(f'{params.cpl["path2save"]}/user_clusters2.csv')

            user_clusters3 = pd.DataFrame({'UserId': user_features['UserId'], 'Community': labels3.tolist()})
            with open(f"{params.cpl['path2save']}/PredUserClusters3.npy", 'wb') as f:
                pickle.dump(labels3, f)
            user_clusters3.to_csv(f'{params.cpl["path2save"]}/user_clusters3.csv')


            return user_clusters, labels
    elif params.cpl['type'] == 'graph_based':
        if params.cpl['method'] == 'louvain':
            from sknetwork.clustering import Louvain
            # louvain = Louvain(resolution=1, n_aggregations=200, shuffle_nodes=True, return_membership=True,
            #                       return_aggregate=True, verbose=True)
            # louvain2 = Louvain(resolution=1, max_agg_iter=200, shuffle_nodes=True, verbose=1)
            louvain = Louvain(resolution=1, shuffle_nodes=True, verbose=1)
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(predicted_features, predicted_features)
            np.save(f'{params.cpl["path2save"]}/adj.npy', similarities)
            cmn.logger.info(f"(size) : ({similarities.size})")
            cmn.logger.info(f"(#zeros) : ({similarities.size - np.count_nonzero(similarities)})")
            adj2 = sparse.csr_matrix(similarities)
            pd.to_pickle(adj2, f'{params.cpl["path2save"]}/adj_sparse.pkl')
            labels = np.asarray(louvain.fit_transform(adj2))
            argmax_indices_dense = np.asarray(np.argmax(labels.tolist().todense(), axis=1).flatten())[0]
            user_clusters = pd.DataFrame({'UserId': user_features['UserId'], 'Community': argmax_indices_dense})
            with open(f"{params.cpl['path2save']}/PredUserClusters.npy", 'wb') as f:
                pickle.dump(labels, f)
            user_clusters.to_csv(f'{params.cpl["path2save"]}/user_clusters.csv')
            return user_clusters, labels

            # np.save(f'{params.cpl["path2save"]}/PredUserClusters.npy', labels)
            # print(labels)
            # pd.DataFrame(labels).to_csv(f'{params.cpl["path2save"]}/user_clusters.csv')
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
    print(embeddings.shape)

    pred_users_similarity = pairwise_kernels(embeddings, metric='cosine', n_jobs=9)
    pred_users_similarity[pred_users_similarity < params.uml['userSimilarityThreshold']] = 0
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

