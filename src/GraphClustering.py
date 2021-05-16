import networkx as nx
import sknetwork as skn
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import logging
from collections import Counter


def ChangeLoc():
    print(os.getcwd())
    run_list = glob.glob('../output/2021*')
    print(run_list[-1])
    os.chdir(run_list[-1]+'/graphs')

def LogFile():
    file_handler = logging.FileHandler("../logfile.log")
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.setLevel(logging.ERROR)
    logger.critical("\nGraphClustering.py:\n")
    return logger

def GraphShow(G,day):
    G = G.subgraph(list(G.nodes)[:500])
    nx.draw(G)#, with_labels=True)
    plt.interactive(False)
    # plt.show(block=True)
    plt.savefig('Graph'+str(day)+'.jpg')
    plt.close()

def ClusterTopicInterest(clusters, usertopicinterests, logger):
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
        logger.critical("Cluster "+str(ci)+" has "+str(len(clusterInterests[ci]))+' users. Topic '+str(topic)+' is the favorite topic for '+str(countpercentage)+ '% of users.')


def GC_main():
    # ChangeLoc()
    logger = LogFile()
    louvain = skn.clustering.Louvain(resolution=1, n_aggregations=200, shuffle_nodes=True, return_membership=True,
                                     return_aggregate=True, verbose=1)
    graphName = glob.glob('*.net')[-1]
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

    logger.critical("Louvain clustering for " + graphName)
    logger.critical(
        "nodes: " + str(graph.number_of_nodes()) + " / edges: " + str(graph.number_of_edges()) + " / isolates: " + str(
            nx.number_of_isolates(graph)))
    logger.critical("Louvain clustering output: " + str(lbls_louvain.max()) + " clusters. " + str(
        len(clusterMembers)) + " of them are multi-user clusters and rest of them (" + str(
        lbls_louvain.max() - len(clusterMembers)) + ") are singleton clusters.\n")
    logger.critical('Length of multi-user clusters: ' + str(clusterMembers) + '\n')
    np.save('../UserClusters.npy', lbls_louvain)
    logger.critical("UserClusters.npy saved.\n")

    # GraphShow(G_t,100)
    # GraphShow(G_t2,101)
    UTIName = glob.glob('../Day*UsersTopicInterests.npy')[-1]
    UTI = np.load(UTIName)
    ClusterTopicInterest(lbls_louvain, UTI, logger)