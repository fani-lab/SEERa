import networkx as nx
import sknetwork as skn
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from cmn import Common as cmn
from collections import Counter


def GraphShow(G,day):
    G = G.subgraph(list(G.nodes)[:500])
    nx.draw(G)#, with_labels=True)
    plt.interactive(False)
    # plt.show(block=True)
    plt.savefig('Graph'+str(day)+'.jpg')
    plt.close()

def ClusterTopicInterest(clusters, usertopicinterests):
    clusterInterests = []
    for i in range(clusters.max()+1):
        clusterInterests.append([])
    for u in range(len(usertopicinterests)):
        cluster = clusters[u]
        clusterInterests[cluster].append(usertopicinterests[u].argmax())
    for ci in range(len(clusterInterests)):
        c = Counter(clusterInterests[ci])
        topic, count = c.most_common()[0]
        countpercentage = (count/len(clusterInterests[ci]))*100
        cmn.logger.info("Cluster "+str(ci)+" has "+str(len(clusterInterests[ci]))+' users. Topic '+str(topic)+' is the favorite topic for '+str(countpercentage)+ '% of users.')


def main(RunId):
    louvain = skn.clustering.Louvain(resolution=1, n_aggregations=200, shuffle_nodes=True, return_membership=True,
                                     return_aggregate=True, verbose=1)
    print(os.getcwd())
    graphName = glob.glob(f'../output/{RunId}/uml/graphs/*.net')[-1]
    print(f'../output/{RunId}/uml/graphs/*.net')
    print(graphName)
    graph = nx.read_gpickle(graphName)
    adj = nx.adj_matrix(graph)
    print('Louvain 1')
    # lbls_louvain = louvain.fit_transform(cosine_adj)
    print('Louvain2')
    lbls_louvain = louvain.fit_transform(adj)
    clusterMembers = []
    for UC in range(lbls_louvain.min(), lbls_louvain.max()):
        UsersinCluster = np.where(lbls_louvain == UC)[0]
        if len(UsersinCluster) == 1:
            break
        else:
            clusterMembers.append(len(UsersinCluster))

    print(lbls_louvain.shape)
    print(lbls_louvain.max())

    cmn.logger.info("Graph Clustering: Louvain clustering for " + graphName)
    cmn.logger.info(
        "nodes: " + str(graph.number_of_nodes()) + " / edges: " + str(graph.number_of_edges()) + " / isolates: " + str(
            nx.number_of_isolates(graph)))
    cmn.logger.info("Graph Clustering: Louvain clustering output: " + str(lbls_louvain.max()) + " clusters. " + str(
        len(clusterMembers)) + " of them are multi-user clusters and rest of them (" + str(
        lbls_louvain.max() - len(clusterMembers)) + ") are singleton clusters.\n")
    cmn.logger.info('Graph Clustering: Length of multi-user clusters: ' + str(clusterMembers) + '\n')
    np.save(f'../output/{RunId}/uml/UserClusters.npy', lbls_louvain)
    cmn.save2excel(lbls_louvain, 'uml/UserClusters')
    cmn.logger.info("Graph Clustering: UserClusters.npy saved.\n")

    # GraphShow(G_t,100)
    # GraphShow(G_t2,101)
    UTIName = glob.glob(f'../output/{RunId}/uml/Day*UsersTopicInterests.npy')[-1]
    UTI = np.load(UTIName)
    ClusterTopicInterest(lbls_louvain, UTI)

